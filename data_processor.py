import random
import re
from typing import Dict, List, Union


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize the input text into a list of tokens
    """
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)


def transform_data(data: Dict[str, str]) -> Dict[str, Union[str, List[str], bool]]:
    """
    Transforms the input data into the required format for training.
    Args:
        data: A list of dictionaries.
            Each dict is expected to contain keys 'text' (str) and 'entities' (List[dict]).
            Each dict in entities is expected to contain:
                - 'entity' key indicating the entity name
                - 'word' key indicating the exact entity string value
    """
    assert all(key in data.keys() for key in ['text', 'entities']), "The data record does not contain all required keys"
    tokens = tokenize_text(data['text'])
    spans = []

    for entity in data['entities']:
        assert all(key in entity for key in ['entity', 'word']), "The data record does not contain all required keys"
        entity_tokens = tokenize_text(entity['word'])
        entity_length = len(entity_tokens)

        # Find the start and end indices of each entity in the tokenized text
        for i in range(len(tokens) - entity_length + 1):
            if tokens[i:i + entity_length] == entity_tokens:
                spans.append([i, i + entity_length - 1, entity['entity']])
                break
    
    return {
        "tokenized_text": tokens,
        "ner": spans,
        "validated": False
    }


def train_test_split(data: List[dict], split_ratio: float = 0.8):
    random.seed(42)
    random.shuffle(data)
    train_data = data[:int(len(data) * split_ratio)]
    test_data = data[int(len(data) * split_ratio):]
    print(f"Training data size: {len(train_data)}, Testing data size: {len(test_data)}")
    return train_data, test_data
