MODEL_PATH="./"
DATA_PATH="./pano_np_0.jpg"
python infer.py  --image_path $DATA_PATH --dataset matterport3d --load_weights_dir $MODEL_PATH 
