from datasets import load_dataset
from typing import Dict, List


def get_hf_data(dataset_name: str) -> List[Dict]:
    data = load_dataset(dataset_name)
    return data['train'].to_pandas().to_dict(orient='records')
