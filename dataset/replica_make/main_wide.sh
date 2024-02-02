# python main_wide.py --part 1 --split train --dist 0.5
# python main_wide.py --part 1 --split train --dist 1.0
# dist=0.25
# python main_wide.py --part 0 --split test --dist $dist

for line in `cat ./baselines.txt`;
do 
    m3d_dist=$line;
    # m3d_dist=0.5 #0.75, 1.0, 0.25
    for part_idx in {1..4};
    do
    # part_idx=2;
        python main_wide.py --part $part_idx --split train --dist $m3d_dist
    done;
    python main_wide.py --part 0 --split test --dist $m3d_dist; #
done;
