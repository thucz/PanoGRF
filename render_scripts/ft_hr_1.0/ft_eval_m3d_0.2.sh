# wget https://download.pytorch.org/models/vgg16-397923af.pth -O ~/.cache/torch/hub/vgg16-397923af.pth
mkdir -p /usr/local/app/.cache/torch/hub/checkpoints/
cp checkpoints/* /usr/local/app/.cache/torch/hub/checkpoints/

# _handle_distort_wloss.yaml
database_name="m3d"
# data_cnt=5 
# for line in `cat ./baselines.txt`;
# do
    # m3d_dist=$line;
    m3d_dist=0.1;
    basedir="/group/30042/ozhengchen/ft_local/NeuRay-spherical-broken-ae-erp+tp/data/render/${database_name}_"$m3d_dist
    echo $m3d_dist;
    # for num in {0..9};
    # do
        num=0;
        name="neuray_ft_m3d_diff_mono_uniform_hr_0.2_id_$num-10000"

        # python # render on fern of the LLFF dataset
        # CUDA_VISIBLE_DEVICES=0 python render.py --cfg configs/train/ft_hr_1.0/neuray_ft_cv_m3d_diff_mono_uniform_0.2.yaml \
        #                 --database_name $database_name \
        #                 --pose_type "eval" \
        #                 --render_type "ft" \
        #                 --data_idx $num \
        #                 --m3d_dist $m3d_dist
        dir_gt=$basedir/"$name-eval-gt"
        dir_pr=$basedir/"$name-eval"
        python eval.py --dir_gt $dir_gt \
                        --dir_pr $dir_pr
    # done;
# done;