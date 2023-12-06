
# _handle_distort_wloss.yaml
m3d_dist=0.25
name="replica_wide";

for num in {0..17};
do
    for cube_id in {0..5};
    do

    # python # render on fern of the LLFF dataset
    CUDA_VISIBLE_DEVICES=0 python render_cubes.py --cfg configs/train/gen_hr_1.0/neuray_gen_cv_erp_mono_stereo_uniform_512x1024.yaml \
                    --database_name $name \
                    --pose_type "inter" \
                    --data_idx $num \
                    --cube_id $cube_id \
                    --m3d_dist $m3d_dist
    done;
done;