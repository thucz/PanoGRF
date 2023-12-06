
# _handle_distort_wloss.yaml
m3d_dist=0.25
for num in {1..3};
do
# num=0;

# python # render on fern of the LLFF dataset
CUDA_VISIBLE_DEVICES=0 python render.py --cfg configs/train/gen_hr_1.0/neuray_gen_cv_erp_mono_stereo_uniform_512x1024.yaml \
                 --database_name "m3d" \
                 --pose_type "inter" \
                 --data_idx $num \
                 --m3d_dist $m3d_dist
done;