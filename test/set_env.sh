## set env on lambda

## 1. conda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh;
sh Miniconda3-latest-Linux-x86_64.sh;

## 2. cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin;
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600;
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb;
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb;
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/;
sudo apt-get update;
sudo apt-get -y install cuda-toolkit-12-4;

export CUDA_HOME=/usr/local/cuda-12.4;
export PATH=$CUDA_HOME/bin:$PATH;
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH;

sudo ln -s /usr/local/cuda-12.4 /usr/local/cuda;
nvcc --version;

## 3. conda env
conda create -c pytorch -c nvidia -n pointcept pytorch torchvision torchaudio pytorch-cuda=12.4 ipython;
source ~/.bashrc;
conda activate pointcept;

### 4. other dependencies
pip install torch-cluster torch-scatter torch-sparse
## or
### pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
### pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
### pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm pyyaml -c conda-forge -y
pip install torch-geometric
pip install flash-attn --no-build-isolation


## 5.spconv and cumm
git clone https://github.com/FindDefinition/cumm; cd ./cumm;git checkout tags/v0.7.11; pip install -e .;cd ..;
python "import cumm"
git clone https://github.com/peterpan83/spconv; cd ./spconv; pip install -e .;cd ..;

#### compile spconv
python -c "import spconv"

## 6. pointcept
git clone https://github.com/peterpan83/Pointcept.git
cd Pointcept/libs/pointops
python setup.py install
#if error occured: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
#pip install setuptools==68

export PYTHONPATH=/home/ubuntu/yanai/Pointcept:$PYTHONPATH






