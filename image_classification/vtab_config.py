config = {
    "cifar": {
        "init_mean": 1.5,
        "init_std": 0.1,
        "scale": 0.1,
        "seed": 14,
        "logger": False
    },
    "caltech101": {
        "init_mean": 0.9,
        "init_std": 0.01,
        "scale": 100,
        "seed": 56,
        "logger": False
    },
    "dtd": {  # Dropout: 0.3
        "init_mean": 1.0,
        "init_std": 0.0,
        "scale": 0.1,
        "seed": 14,
        "logger": False
    },
    "oxford_flowers102": { # Dropout: 0.3
        "init_mean": 1.0,
        "init_std": 0.02,
        "scale": 10.0,
        "seed": 50,
        "logger": False
    },
    "oxford_iiit_pet": { # Dropout: 0.3
        "init_mean": 1.2,
        "init_std": 0.06,
        "scale": 1.0,
        "seed": 93,
        "logger": False
    },
    "svhn": {
        "init_mean": 1.0,
        "init_std": 0.05,
        "scale": 100,
        "seed": 14,
        "logger": False
    },
    "sun397": { # Dropout: 0.3
        "init_mean": 1.35,
        "init_std": 0.06, 
        "scale": 1.0,
        "seed": 43, 
        "logger": False
    },
    "patch_camelyon": {
        "init_mean": 1.0,
        "init_std": 0.0,
        "scale": 10,
        "seed": 89,
        "logger": False
    },
    "eurosat": {
        "init_mean": 1.08,
        "init_std": 0.028,
        "scale": 10,
        "seed": 32,
        "logger": False
    },
    "resisc45": {
        "init_mean": 1.16,
        "init_std": 0.03,
        "scale": 10,
        "seed": 28,
        "logger": False
    },
    "diabetic_retinopathy": { # dropout 0.3
        "init_mean": 1.0,
        "init_std": 0.0,
        "scale": 0.1,
        "seed": 81,
        "logger": False
    },
    "clevr_count": {
        "init_mean": 1.0,   
        "init_std": 0.0,    
        "scale": 5,         
        "seed": 44,          
        "logger": False
    },
    "clevr_dist": {   # dropout 0.3
        "init_mean": 1.0,
        "init_std": 0.0,
        "scale": 2.5,
        "seed": 25,
        "logger": False
    },
    "dmlab": {
        "init_mean": 1.0,
        "init_std": 0.0,
        "scale": 10,
        "seed": 72,
        "logger": False
    },
    "kitti": {
        "init_mean": 1.0,
        "init_std": 0.0,
        "scale": 5,
        "seed": 31,
        "logger": False
    },
    "dsprites_loc": {
        "init_mean": 1.0,
        "init_std": 0.0,
        "scale": 50,
        "seed": 12,
        "logger": False
    },
    "dsprites_ori": {        # Dropout: 0.3  
        "init_mean": 1.3,
        "init_std": 0.07,
        "scale": 1.0,
        "seed": 79,
        "logger": False
    },
    "smallnorb_azi": {
        "init_mean": 1.0,
        "init_std": 0.0,
        "scale": 100,
        "seed": 67,
        "logger": False
    },
    "smallnorb_ele": { #  Dropout: 0.3
        "init_mean": 1.0,
        "init_std": 0.0,
        "scale": 10.0,
        "seed": 30,
        "logger": False
    }
}