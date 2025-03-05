import torch
import torch.nn.functional as F
import os
import json
import evaluate
import numpy as np
import wandb
import argparse

from huggingface_hub import login as hf_login
from wandb import login as wandb_login
from transformers import Trainer, TrainingArguments
from model import tokenizer, model, tokenized_datasets
from dotenv import load_dotenv
from pprint import pprint

from config import Config
from utils import GCSUploadCallback, EarlyStoppingTrainingLossCallback, get_memory_usage

load_dotenv()

hf_key = os.environ["HUGGING_FACE_API_KEY"]
wandb_key = os.environ["WANDB_API_KEY"]


hf_login(token=hf_key)
wandb_login(key=wandb_key)

wandb.init(project=Config.PROJECT_NAME)
# Load multiple metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(logits)
    print(labels)
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],  # type: ignore
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],  # type: ignore
    }


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        device = "cpu"

        if isinstance(model, torch.nn.DataParallel):
            print(f"devices: {model.device_ids}")
            device = model.module.device

        print(f"model device: {device}")

        # inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop("labels")

        print(f'inputs shape: {inputs["input_ids"].shape}')

        mask_token_id = tokenizer.mask_token_id
        outputs = model(**inputs)
        logits = outputs.logits

        device = logits.device
        labels = labels.to(device)

        print(f"logits device: {device}")

        # get mask_token_ids of all inputs
        mask_token_indices = torch.where(inputs["input_ids"] == mask_token_id)

        logits = logits[mask_token_indices[0], mask_token_indices[1], :]

        loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train(bkt_upload=True):

    training_args = TrainingArguments(
        run_name=Config.PROJECT_NAME,
        output_dir=Config.OUTPUT_DIR,  # output directory
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
        learning_rate=5e-5,
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=300,
        save_steps=300,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        bf16=True
        # optim="adamw_bnb_8bit",
        # ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    es_callback = EarlyStoppingTrainingLossCallback(patience=3)
    gcs_callback = GCSUploadCallback(bkt_upload=bkt_upload)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics,
        callbacks=[es_callback, gcs_callback],
    )

    # initiating the training
    trainer.train()

    prediction_outputs = trainer.predict(test_dataset=tokenized_datasets["test"])  # type: ignore
    best_checkpoint = trainer.state.best_model_checkpoint

    print("predictions")
    pprint(prediction_outputs.metrics)

    with open("best_checkpoint.json", "w") as f:
        json.dump(obj={"checkpoint": best_checkpoint}, fp=f)
    print(f"Best Checkpoint: {best_checkpoint} Saved.")



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="User Info CLI")
    parser.add_argument("-u", "--upload", action="store_true", help="Enable uploads")
    args = parser.parse_args()
    
    bkt_upload = False
    if args.upload:
        print(f'bucket upload enabled')
        bkt_upload=True
    else:
        print(f'bucket upload disabled')
    
    train(bkt_upload=bkt_upload)


