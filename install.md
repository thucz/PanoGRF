

# We take anaconda3 for example
```
conda create -n panogrf python=3.7
conda activate panogrf

# pip install pytorch for CUDA11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# requirements
cd PanoGRF
pip install -r requirements.txt
```

# install habitat
<!-- pip install protobuf==3.20.1 proglog decorator msgpack simplejson click distro progress billiard einops kornia -->

## habitat-lab

use pip:
```
pip install git+https://github.com/facebookresearch/habitat-lab.git@v0.2.2
```

If you fail to install habitat-lab when using pip, you can also install it from source code. Refer to [official habitat repository](https://github.com/facebookresearch/habitat-lab)


## habitat-sim
```
conda install habitat-sim=0.2.2 headless -c conda-forge -c aihabitat -y
```
This step takes a long time for me.

