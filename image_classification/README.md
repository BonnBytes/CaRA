CP form of LoRA, evaluation on VTAB-1K dataset.

Inspirations:

1. FacT: [Code](https://github.com/JieShibo/PETL-ViT/tree/main/FacT)
2. GLoRA: [Code](https://github.com/Arnav0400/ViT-Slim/tree/master/GLoRA)
3. NOAH: [Code](https://github.com/ZhangYuanhan-AI/NOAH)


### Set up:
Follow the steps as described over at `https://github.com/JieShibo/PETL-ViT/tree/main/FacT`.
Additionally run
``` bash
pip install tqdm
pip install tensorly
```

### Running experiments
To run an experiment type:

``` bash
python fact_cp.py --dataset 'dataset_name'
```

Choose a dataset name from i.e. '

``` bash
caltech101 
clevr_dist           
dsprites_loc  
eurosat
oxford_iiit_pet 
smallnorb_azi
cifar      
diabetic_retinopathy 
dsprites_ori
kitti
patch_camelyon
smallnorb_ele
clevr_count
dmlab                
dtd
oxford_flowers102
resisc45
sun397
svhn
```
