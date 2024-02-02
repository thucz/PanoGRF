input_path='/mnt/data1/chenzheng/tmp/residential/ricoh_mini'
scene_number=1
output_path=/mnt/data1/chenzheng/tmp/residential/ricoh_mini/${scene_number}_perspective_all/images
pose_out_path=/mnt/data1/chenzheng/tmp/residential/ricoh_mini/${scene_number}_perspective_all/poses_to_sphere

scene_dir=/mnt/data1/chenzheng/tmp/residential/ricoh_mini/${scene_number}_perspective_all #save all the perspective data: all.t7

fov=90
# subid_dict = {
#     (90, 0):   0,
#     (0, 0):    1,
#     (0, -90):  2,
#     (0, -180): 3,
#     (0, -270): 4,
#     (-90, 0):  5,
# }

# phi=90
# theta=0

python3 residential_preprocess/residential_handle.py \
        -o=$output_path \
        -i=$input_path \
        -s=$scene_dir \
        --pose_out_path=$pose_out_path \
        --fov $fov \
        --scene_number $scene_number
        # --phi $phi \
        # --theta $theta \
