# preprocess Replica


## download the original Replica Dataset
Refer to the official [repo](https://github.com/facebookresearch/Replica-Dataset)


## prepare
```
# download synsin to get the scene_episodes of replica.
git clone https://github.com/facebookresearch/synsin.git

```

remember to set the following directory path to your own in `class Options` of `main_wide.py`:

`self.scenes_dir`: Replica data storage directory

`self.basedir`: the base directory where you put synsin and mp3d.yaml

`self.episodes_dir`: the specific directory where you put scene episodes.

`final_save_dir`: the directory to save processed data


Replica data structure:
```
${self.scenes_dir}
     | -- apartment_0
             | -- glass.sur
             | -- mesh.ply
             ...
     | -- apartment_1
             | -- ...
             | -- ...
```
## preprocess
run script to preprocess the original data to get 360-degree replica data.
`bash main_wide.sh`

