# Usages

## GCN

### ogbl-ppa
```bash
python -u main.py --dataset ogbl-ppa --eval_method Hits@100 --positional --epochs 20
```

### ogbl-ddi
```bash
python -u main.py --dataset ogbl-ddi --eval_method Hits@20 --lr 0.005 --num_layers 2 --positional --dropout 0.5
```

## SEAL

### ogbl-ppa
```bash
python -u main.py --dataset ogbl-ppa --eval_method Hits@100 --epochs 20 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 0.0001 --train_percent 5 --eval_steps 5 --use_feature --dynamic_train --seal --runs 1
```
### ogbl-ddi
```bash
python -u main.py --dataset ogbl-ddi --eval_method Hits@20 --epochs 10 --num_layers 3 --hidden_channels 32 --batch_size 32 --lr 0.0001 --train_percent 1 --ratio_per_hop 0.2 --seal --runs 1
```


## Note
Consider launching the commands 10 times with `--runs 1` instead of once with `--runs 10`
(which is the default, i.e. what you get if you don't specify it too) to exploit parallelism.