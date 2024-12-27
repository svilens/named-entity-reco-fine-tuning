# Overview

This repository is an example for fine-tuning models on named entity recognition task.

The base model used is [knowledgator/gliner-multitask-large-v0.5]('https://huggingface.co/knowledgator/gliner-multitask-large-v0.5').

The fine-tuning data is expected to be in the following format:
```json
[
    {
        "text": "...",  // this is the input example
        "entities": [
            {
                "entity": "...",  // this refers to the entity (label) name in the input
                "word": "..."  // this refers to the corresponding exact entity (label) string in the input
            },
            ...
        ]
    },
    ...
]
```
The model will be fine-tuned on the task to get the "text" as input and return the "entities" as output.

The process is started by:

```
python train.py
```
