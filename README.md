Best Results:

OGBL-DDI Test Score Avg. 0.2625 +- 0.0356, with a training time of 360 seconds per run. 

To replicate, run

python -u main.py --dataset ogbl-ddi --eval_method Hits@20 --epochs 10 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 0.0001 --l2_strength 0.001 --train_percent 1 --eval_steps 1 --seal --runs 10 --dynamic_train --ratio_per_hop .2  --num_hops 1
