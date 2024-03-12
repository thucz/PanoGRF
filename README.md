# PanoGRF

This is the code release for our NeurIPS2023 paper, PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas
## [Project Page](https://thucz.github.io/PanoGRF/)|[Arxiv](https://arxiv.org/abs/2306.01531)

## Update:
2024.2.2 upload the preprocess files for Replica and Residential. See the README files for [replica](./dataset/replica_make/README.md) and [Residential](./dataset/residential_make/README.md)


## Citation
If you find this repo useful, please give me a star and cite this paper:
```
@article{chen2023panogrf,
  title={PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas},
  author={Chen, Zheng and Cao, Yan-Pei and Guo, Yuan-Chen and Wang, Chen and Shan, Ying and Zhang, Song-Hai},
  journal={arXiv preprint arXiv:2306.01531},
  year={2023}
}
```

## Environment
Refer to [installation guidance](./docs/install.md)



## Dataset 

#### Download
We download Matterport3D following [SynSin](https://github.com/facebookresearch/synsin/blob/main/MP3D.md).

Please fill and sign the [Terms of Use](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) agreement form and send it to matterport3d@googlegroups.com to request access to the dataset.

The offical download script (`download_mp.py`) will be given in the reply email after your request is approved.

The full MP3D dataset for use with Habitat can be downloaded using the official [Matterport3D](https://niessner.github.io/Matterport/) download script as follows: python download_mp.py --task habitat -o path/to/download/. Note that this download script requires python 2.7 to run.


#### Dataset Path Configuration
You should change the name of all the saving directories in the config files according to your directories.

You should also revise the data directory `opts.scenes_dir` according to your download directory which stored `mp3d` in the following files:
```
data_readers/habitat_data_neuray_ft.py
data_readers/habitat_data_neuray.py
```


The data for `opts.scene_dir` is organized as:
```
<opts.scene_dir>
|-- mp3d 
    |-- 1LXtFkjw3qL # scene_name
            |-- 1LXtFkjw3qL_semantics.ply
            |-- 1LXtFkjw3qL.glb
            |-- 1LXtFkjw3qL.house
            |-- 1LXtFkjw3qL.navmesh            
    |-- 1pXnuDYAj8r
            |-- ...
    |-- ...
```
## pretrained model
The pretrained models of 360-degree Monocular Net, 360-degree MVSNet and general renderer(two-views trained under 1.0m camera baseline) can be found in [GoogleDrive](https://drive.google.com/drive/folders/14RTKIsmQVuBc-b_z8f2iCb0cjc6UdVBN?usp=sharing)

## Depth Training
### Monocular depth finetuning:
Download the pretrained model on Matterport3D from [UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion) and put it under the directory `load_weights_dir` in config file `configs/train/depth/m3d_mono.yaml`.

Training the monocular depth network as follows:
```
bash train_scripts/train_depth/train_monodepth.sh
```
### Multi-view stereo network training with monocular depth prior:
set `DNET_ckpt` in `configs/train/depth/m3d_mono.yaml` to the path of monocular depth model in last step. 
```
bash train_scripts/train_depth/train_mvs.sh
```
## General renderer training:
The speed of rendering training data (textured mesh) online with habitat is tolerable in depth training.
But it is quite slow for training general renderer. So I used lmdb to pre-rendering and save the data here.

Data preprocessing using lmdb: 

```
# preprocessing training data:
python lmdb_rw_render/lmdb_write_render.py --cfg configs/data/train_data_render_512x1024.yaml
# please preprocess val/test data similarly.
```

This step takes too much storage space. If the saved data is too large for you, try to reduce `total_cnt` in data-preprocessing config file and revise it in the training config files correspondingly.

```
bash train_scripts/gen_hr_1.0/gen_mono_stereo_uniform_512x1024.sh
```
## render & eval with pretrained models:
In configuration file `configs/train/gen_hr_1.0/neuray_gen_cv_erp_mono_stereo_uniform_512x1024.yaml`,
you need to revise `DNET_ckpt` to pretrained monocular depth model path: `habitat_monodepth/checkpoint_100000.pt`
`mvsnet_pretrained_path` to pretrained MVS depth model path `habitat_mvs/checkpoint_100000.pt``

run `mkdir -p data/neuray_gen_erp_1.0_mono_stereo_uniform_512x1024`,put pretrained renderer model `general_renderer/model.pth` into `data/neuray_gen_erp_1.0_mono_stereo_uniform_512x1024`

Then run the following command to get renderer results.
```
bash render_scripts/gen_hr_1.0/gen_eval_m3d.sh 
```
## Todo List
- [ ] multi-view training
- [ ] fine-tune training
- [x] Dataset: Replica
- [x] Dataset: Residential
- [ ] clean up unnessary codes
- [ ] ......

## Acknowledgements
Within this repository, I have utilized code and datasets from various sources. I express my gratitude to all the authors who have generously shared their valuable resources, enabling me to build upon their work:
* [SOMSI](https://github.com/tedyhabtegebrial/SoftOcclusionMSI)
* [OmniSyn](https://github.com/AugmentariumLab/omnisyn)
* [MaGNet](https://github.com/baegwangbin/MaGNet)
* [UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)
* [SynSin](https://github.com/facebookresearch/synsin/tree/main/data)
* [Matterport3D](https://niessner.github.io/Matterport/)
* [Replica](https://github.com/facebookresearch/Replica-Dataset)
* [Residential](https://github.com/tedyhabtegebrial/SoftOcclusionMSI)






