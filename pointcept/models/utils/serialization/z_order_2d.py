## adapted for 2D data from pointcept by Yanqun Pan (31 Jan 2025)

import torch
from typing import Optional, Union


class KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r64 = torch.arange(64, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self.xy2key(r256, zero, 8),
                self.xy2key(zero, r256, 8)
            )
        }
        self._decode = {device: self.key2xy(r64, 6)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xy2key(self, x, y, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (i + 1))
                | ((y & mask) << (i + 0))
            )
        return key

    def key2xy(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (2 * i + 1))) >> (i + 1))
            y = y | ((key & (1 << (2 * i + 0))) >> (i + 0))
        return x, y


_key_lut = KeyLUT()

def xy2key(
    x: torch.Tensor,
    y: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    r"""Encodes :attr:`x`, :attr:`y`, coordinates to the shuffled keys
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      x (torch.Tensor): The x coordinate.
      y (torch.Tensor): The y coordinate.
      b (torch.Tensor or int): The batch index of the coordinates, and should be
          smaller than 32768. If :attr:`b` is :obj:`torch.Tensor`, the size of
          :attr:`b` must be the same as :attr:`x`, :attr:`y`, and :attr:`z`.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    EX, EY = _key_lut.encode_lut(x.device)
    x, y = x.long(), y.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask]
        key = key16 << 16 | key

    if b is not None:
        b = b.long()
        key = b << 32 | key

    return key


def key2xy(key: torch.Tensor, depth: int = 16):
    r"""Decodes the shuffled key to :attr:`x`, :attr:`y`
    and the batch index based on pre-computed look up tables.

    Args:
      key (torch.Tensor): The shuffled key.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    DX, DY = _key_lut.decode_lut(key.device)
    x, y= torch.zeros_like(key), torch.zeros_like(key)

    b = key >> 32
    key = key & ((1 << 32) - 1)

    n = (depth + 2) // 3
    for i in range(n):
        k = key >> (i * 6) & 63
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
    return x, y, b