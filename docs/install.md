

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

# EGL+OpenGL
Make sure you have installed EGL + OpenGL in your environment. Check out by [egl-example](https://github.com/erwincoumans/egl_example).

If you haven't installed egl, egl installation procedure from docker file is like [this](./egl_dockerfile).

# Habitat
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



# Others
if your habitat environment cannot work, you may check your environment via `requirements_freeze.txt`

```
...
-e git+https://github.com/facebookresearch/habitat-lab.git@0f454f62e41050bc90ca468c62db35d7484923ff#egg=habitat
-e git+https://github.com/facebookresearch/habitat-lab.git@afe4058a7f8aa5ab71a133575cdaa79f0308af6a#egg=habitat_lab&subdirectory=habitat-lab
habitat-sim==0.2.2
...
```