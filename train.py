from data_loader import get_hf_data
from data_processor import train_test_split, transform_data

from gliner import GLiNER
from gliner.data_processing import GLiNERDataset
from gliner.data_processing.collator import DataCollatorWithPadding
from gliner.training import Trainer, TrainingArguments
import os
import torch
from typing import List


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def create_models_dir(path: str):
    if not os.path.exists(path):
        print(f'Creating directory {path}')
        os.makedirs(path)


def setup_mlflow(experment_name: str = 'gliner', uri: str = "http://localhost:5000"):
    os.environ["MLFLOW_TRACIKNG_URI"] = uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experment_name
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"


def train_model(
    model_name: str,
    data: List[dict],
    new_model_name: str = 'gliner-fine-tuned',
    split_ratio: float = 0.8,
    learning_rate: float = 5e-6,
    weight_decay: float = 0.05,
    batch_size: int = 8,
    epochs: int = 100,
    compile_model: bool = False,
    device: str = 'cuda',
    output_dir: str = './gliner_model',
    report_to: str = 'none'
):  
    if report_to == 'mlflow':
        setup_mlflow(experment_name=new_model_name)
    
    create_models_dir()

    print(f"Loading model using device {device}")
    model = GLiNER.from_pretrained(model_name, device='cpu')
    train_data, test_data = train_test_split(data, split_ratio)
    train_dataset = GLiNERDataset(examples=train_data, config=model.config, data_processor=model.data_processor)
    test_dataset = GLiNERDataset(examples=test_data, config=model.config, data_processor=model.data_processor)
    data_collator = DataCollatorWithPadding(model.config)

    if compile_model:
        print("Compiling model for faster training...")
        torch.set_float32_matmul_precision('high')
        model.to(device)
        model.compile_for_training()
    else:
        model.to(device)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        others_lr=learning_rate,
        others_weight_decay=weight_decay,
        lr_scheduler_type='linear',
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_steps=1000,
        save_total_limit=10,
        dataloader_num_workers=8,
        use_cpu=False,
        report_to=report_to
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/{new_model_name}")
    print('Training completed successfully!')

if __name__ == '__main__':
    data = get_hf_data(dataset_name='svilens/auto-tires-ner')
    train_model(
        model_name='knowledgator/gliner-multitask-large-v0.5',
        new_model_name='gliner-multitask-large-v0.5-custom',
        data=[transform_data(d) for d in data]
    )