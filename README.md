## Instructions for the paper

---

1. :blush: init
    - `mkdir best_model predicts`
    
    - `mkdir ./data/ED/comet` 
   
    - download [comet checkpoint](https://github.com/allenai/comet-atomic-2020) to **comet** directory.
    - `wget http://nlp.stanford.edu/data/glove.6B.zip` to `./vectors` . 
---
result: 

<img src='./cache_files/comet_dir.png' width=100/>
<img src='./cache_files/vectors.png' width=100/>

---

2. :fire: code params explanation
   
| params        | instruction   |  
| --------   | -----:  |
| model    | model type available contains **trans,mult,empdg,mime,moel,kemp,cem,emf** |  
| code_check        |   **store_strue** for fast check program runnable in ur machine  |  
|devices | value passed to `os['CUDA_VISIBLE_DEVICE']` |
|mode| **train_only,train_and_test,test_only** indicates run partial or whole process| 
|max_epoch|max epochs to train model|
|emotion_emb_type|**origin,coarse,contrastive** indicates different emotion embedings,details see paper|
| batch_size       |    batch size for train,valid,test    | 

3. :dog: run experiment

    >* `nohup python train.py --model trans --mode train_and_test --batch_size 32 --max_epoch 128 --devices 0 >trans.log&` **trans** training
    >* `nohup python train.py --model mult --mode train_and_test --batch_size 32 --max_epoch 128 --devices 1 >mult.log&` **mult** training
    >* `nohup python train.py --model empdg --mode train_and_test --batch_size 32 --max_epoch 128 --devices 2 >empdg.log&` **empdg** training
    >* `nohup python train.py --model mime --mode train_and_test --batch_size 32 --max_epoch 128 --devices 3 >mime.log&` **mime** training
    >* `nohup python train.py --model moel --mode train_and_test --batch_size 32 --max_epoch 128 --devices 4 >moel.log&` **moel** training
    >* `nohup python train.py --model cem --mode train_and_test --batch_size 32 --max_epoch 128 --devices 5 >cem.log&` **cem** training
    >* `nohup python train.py --model kemp --mode train_and_test --batch_size 32 --max_epoch 128 --devices 6 >kemp.log&` **kemp** training
4. :mag_right: todo list
   - [ ] run all models
   - [ ] add more params to control
   - [ ] run on other datasets