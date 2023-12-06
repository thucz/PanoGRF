# wget https://download.pytorch.org/models/vgg16-397923af.pth -O ~/.cache/torch/hub/vgg16-397923af.pth
# mkdir -p /usr/local/app/.cache/torch/hub/checkpoints/
# cp checkpoints/* /usr/local/app/.cache/torch/hub/checkpoints/

# _handle_distort_wloss.yaml
database_name="m3d"
total_step=100000
name="neuray_gen_erp_1.0_mono_stereo_uniform_512x1024_mv-"$total_step
# data_cnt=5 
# for line in `cat ./baselines.txt`;
# do
    # m3d_dist=$line;
    m3d_dist=0.5;
    basedir="/group/30042/ozhengchen/ft_local/NeuRay-spherical-broken-ae-erp+tp/data/render/${database_name}_"$m3d_dist
    echo $m3d_dist;
    # for num in {0..9};
    # do
        num=0;
        # python # render on fern of the LLFF dataset
        CUDA_VISIBLE_DEVICES=0 python render_mv.py --cfg configs/train/gen_hr_1.0/neuray_gen_cv_erp_mono_stereo_uniform_512x1024_mv.yaml \
                        --database_name $database_name \
                        --pose_type "eval" \
                        --data_idx $num \
                        --m3d_dist $m3d_dist
        dir_gt=$basedir/"$name-eval-$num-gt"
        dir_pr=$basedir/"$name-eval-$num"
        python eval.py --dir_gt $dir_gt \
                        --dir_pr $dir_pr
    # done;
# done;