database_name="m3d"
name="neuray_gen_erp_1.0_mono_stereo_uniform_512x1024-100000"
m3d_dist=0.1;
echo $m3d_dist;
num=0;

CUDA_VISIBLE_DEVICES=0 python render.py --cfg configs/train/gen_hr_1.0/neuray_gen_cv_erp_mono_stereo_uniform_512x1024.yaml \
                --database_name $database_name \
                --pose_type "eval" \
                --data_idx $num \
                --m3d_dist $m3d_dist
        