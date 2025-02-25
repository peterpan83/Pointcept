import numpy as np
import torch
import os
from tqdm import tqdm


def ohlwiler(data_dir, target_dir, twoD = True, split_ratio = (0.7, 0.2, 0.1)):
    '''
    read the csv data and convert it into multipy npy files that store the coords, and segment
    Args:
        data_dir: contains the csv files
        target_dir: include train, val and test
        twoD: if ture, both Along_Track_Dist and Geoid_Corrected_Ortho_Height are saved as coord, otherwise, only the Along_Track_Dist is saved.
        split_ratio: ratio for train, val and test
    Returns:
    '''
    import glob, os, pandas as pd

    def write_npy(csvf, target = 'train'):
        basename = os.path.basename(csvf).replace('.csv', '')
        _dir = os.path.join(target_dir, target, basename)
        os.makedirs(_dir, exist_ok=True)
        df = pd.read_csv(csvf)
        if twoD:
            coords = df[['Along_Track_Dist', 'Geoid_Corrected_Ortho_Height']].to_numpy().astype(np.float32)
        else:
            coords = df[['Along_Track_Dist']].to_numpy().astype(np.float32)

        segment = df['Manual_Label'].to_numpy(np.int64)
        segment[segment == 40] = 1
        segment[segment == 41] = 2

        np.save(os.path.join(_dir, 'height.npy'), df[['Geoid_Corrected_Ortho_Height']].to_numpy().astype(np.float32))
        np.save(os.path.join(_dir, 'coord.npy'), coords)
        np.save(os.path.join(_dir, 'segment.npy'), segment)

    csvfs = glob.glob(os.path.join(data_dir, '*.csv'))
    df = pd.DataFrame({'fname':csvfs}).sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    train_index = int(split_ratio[0] * n)
    val_index = train_index + int(split_ratio[1] * n)

    train_df, val_df, test_df = df.iloc[:train_index], df.iloc[train_index:val_index], df.iloc[val_index:]
    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        write_npy(row['fname'], 'train')
    for i, row in tqdm(val_df.iterrows(), total=len(val_df)):
        write_npy(row['fname'], 'val')
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        write_npy(row['fname'], 'test')



def seperate_segment(conf, save_to):
    ## grid and segment
    # conf = '/mnt/c/Users/pany0/WorkSpace/open_source/PointceptLocal/Pointcept/configs/icesat2depth/semseg-pt-v3m1-0-base.py'

    from pointcept.datasets import build_dataset, point_collate_fn, collate_fn

    from pointcept.engines.defaults import (
        default_argument_parser,
        default_config_parser,
        default_setup,
    )

    cfg = default_config_parser(conf, options={})

    for target in ['train', 'val', 'test']:
        dataset = build_dataset(cfg.data[target])
        num_dataset = len(dataset)

        for i, data in enumerate(tqdm(dataset, total=num_dataset, desc=target)):
            if i == num_dataset:
                break
            write_npy(data, save_to, target=target)


def write_npy(data,save_to, target = 'train'):
    offsets = np.insert(data['offset'].numpy(), 0,0)
    path = os.path.join(save_to, target, data['name'])

    for i, (left, right) in enumerate(zip(offsets[0:-1], offsets[1:])):
        for key in data.keys():
            if isinstance(data[key], str) or key=='offset':
                continue

            dir_name = f'{path}_{i}'
            os.makedirs(dir_name, exist_ok=True)

            np.save(os.path.join(dir_name, f'{key}.npy'), data[key][left:right].numpy())

    return 0



if __name__ == '__main__':
    # ohlwiler(data_dir='/mnt/e/Projects/ICESAT-2_Bathymetry/Ohlwiler_Data/Updated_CSVs/Updated_CSVs/',
    #          target_dir='/mnt/e/Projects/ICESAT-2_Bathymetry/OhlwilerDataset2D')
    conf = '/mnt/c/Users/pany0/WorkSpace/open_source/PointceptLocal/Pointcept/configs/icesat2depth/semseg-pt-v3m1-0-base_preprocess.py'
    seperate_segment(conf, save_to='/mnt/e/Projects/ICESAT-2_Bathymetry/OhlwilerDataset2D_Seg')




