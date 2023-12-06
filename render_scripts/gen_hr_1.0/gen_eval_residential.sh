database_name="residential"
# name="final_neuray_gen_train_erp_1.0_mono_stereo_uniform_uncert-100000"
name="neuray_gen_erp_1.0_mono_stereo_uniform_512x1024-100000"
for num in {0..2};
do
    # num=1;
    m3d_dist=0.5;
    basedir="/group/30042/ozhengchen/ft_local/NeuRay-spherical-broken-ae-erp+tp/data/render/${database_name}_"$m3d_dist

    CUDA_VISIBLE_DEVICES=0 python render.py --cfg configs/train/gen_hr_1.0/neuray_gen_cv_erp_mono_stereo_uniform_512x1024.yaml \
                    --database_name $database_name \
                    --pose_type "eval" \
                    --data_idx $num \
                    --m3d_dist $m3d_dist
    dir_gt=$basedir/"$name-eval-$num-gt"
    dir_pr=$basedir/"$name-eval-$num"
    python eval.py --dir_gt $dir_gt \
                    --dir_pr $dir_pr
done;