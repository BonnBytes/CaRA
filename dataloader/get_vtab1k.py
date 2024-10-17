import cifar
import svhn
from base import compose_preprocess_fn
from registry import Registry 
import os
import os.path as osp
from PIL import Image

dataset_config = [
    ['cifar', dict(num_classes=100)],
    ['svhn', dict()],
]

import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

data_root = osp.expanduser('../data')

for dataset_name, dataset_params in dataset_config:
    dataset_postfix = dataset_params.pop('dataset_postfix', None)
    data_cls = Registry.lookup(f'data.{dataset_name}')(**dataset_params)
    
    if dataset_postfix is not None:
        dataset_name = dataset_name + '_' + dataset_postfix
    
    os.makedirs(f'{data_root}/{dataset_name}', exist_ok=True)
    os.makedirs(f'{data_root}/{dataset_name}/images', exist_ok=True)
    
    print(f'{dataset_name} started.')
    for split_name in ['train800', 'val200', 'test', 'train800val200']:
        data = data_cls._get_dataset_split(split_name=split_name, shuffle_files=False)
        base_preprocess_fn = compose_preprocess_fn(data_cls._image_decoder, data_cls._base_preprocess_fn)
        data = data.map(base_preprocess_fn, data_cls._num_preprocessing_threads)
        
        os.makedirs(f'{data_root}/{dataset_name}/images/{split_name}', exist_ok=True)
        
        with open(f'{data_root}/{dataset_name}/{split_name}.txt', 'w') as f:
            for i, item in enumerate(data):
                image_path = f'images/{split_name}/{i:06d}.jpg'
                label = item['label'].numpy().item()
                f.write(f'{image_path} {label}\n')
                
                image = item['image'].numpy()
                image = Image.fromarray(image)
                image.save(f'{data_root}/{dataset_name}/{image_path}')
    print(f'{dataset_name} is done.')
