import os
import requests
import tqdm
import zipfile
from dataclasses import dataclass

@dataclass
class DatasetMeta:
    name: str
    url: str

DATASETS_METAS = {
    'fashion-mnist': DatasetMeta(
        name='Fashion-MNIST',
        url='https://www.kaggle.com/api/v1/datasets/download/andhikawb/fashion-mnist-png'
    ),
    'titanic': DatasetMeta(
        name='Titanic',
        url='https://www.kaggle.com/api/v1/datasets/download/brendan45774/test-file'
    )
}

SAVE_DIR = './datasets'

def download_dataset(dataset_name):
    if dataset_name not in DATASETS_METAS:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_URLS.")

    dataset_meta = DATASETS_METAS[dataset_name]
    filename = dataset_meta.name + '.zip'
    filepath = os.path.join(SAVE_DIR, filename)
    extract_path = os.path.join(SAVE_DIR, dataset_name)

    response = requests.get(dataset_meta.url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    if os.path.exists(filepath):
        print(f"Dataset {dataset_name} already downloaded.")
    else:
        with open(filepath, 'wb') as f, tqdm.tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    
        print(f"Dataset {dataset_name} downloaded to {filepath}.")

    if os.path.exists(extract_path):
        print(f"Dataset {dataset_name} already extracted.")
    else:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Dataset {dataset_name} extracted to {SAVE_DIR}/{dataset_name}.")

    
