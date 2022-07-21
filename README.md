## pretrain cmdline

---
### cem emotion tolerance with 4 gpus
>* `nohup python train.py --model cem --emotion_emb_type tolerance --devices 0,1,2,3 > cem_tolerance.log &`

### cem emotion origin with 4 gpus
>* `nohup python train.py --model cem --emotion_emb_type origin --devices 0,1,2,3 > cem_origin.log &`

### cem emotion order with 4 gpus
>* `nohup python train.py --model cem --emotion_emb_type order --devices 0,1,2,3 > cem_order.log &`

### cem emotion random with 4 gpus
>* `nohup python train.py --model cem --emotion_emb_type order --devices 4,5,6,7 > cem_random.log &`

### empdg emotion origin with 4 gpus
>* `nohup python train.py --model empdg --devices 4,5,6,7 > empdg_origin.log &`

### moel emotion origin with 4 gpus
>* `nohup python train.py --model moel --devices 4,5,6,7 > moel_origin.log &`

### trans emotion origin with 4 gpus
>* `nohup python train.py --model trans --devices 4,5,6,7 > trans_origin.log &`