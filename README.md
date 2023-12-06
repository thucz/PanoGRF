# PanoGRF (Under Construction)

This is the code release for our NeurIPS2023 paper, PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas

## [Arxiv](https://arxiv.org/abs/2306.01531)

## Dataset
We download Matterport3D following [SynSin](https://github.com/facebookresearch/synsin/blob/main/MP3D.md).

## Environment
Refer to [installation guidance](./install.md)

## Depth Training

You should change the name of all the saving directories in the config files according to your directories.

You should also revise the data directory `opts.scenes_dir` according to your download directory which stored `mp3d` in the following files.

```
data_readers/habitat_data_neuray_ft.py

data_readers/habitat_data_neuray.py
```
### Monocular depth finetuning:
Download the pretrained model on Matterport3D from [UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion) and put it under the directory `./UniFuse-Unidirectional-Fusion/UniFuse`.

Training the monocular depth network as follows:
```
bash train_depth_scripts/train_monodepth.sh
```

### Multi-view stereo network training:
```
bash train_depth_scripts/train_mvs.sh
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
## render & eval:
```
bash render_scripts/gen_hr_1.0/gen_eval_m3d.sh
```
<!-- ## Todo List

- [ ] Dataset: Replica
- [ ] Dataset: Residential
- [ ] multi-view training & evaluation code
- [ ] code explanation & clean up unnessary codes
- [ ] ...... -->

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




## Citation

```
@article{chen2023panogrf,
  title={PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas},
  author={Chen, Zheng and Cao, Yan-Pei and Guo, Yuan-Chen and Wang, Chen and Shan, Ying and Zhang, Song-Hai},
  journal={arXiv preprint arXiv:2306.01531},
  year={2023}
}
```

