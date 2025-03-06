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
from model import model, tokenize
from dotenv import load_dotenv
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score
from config import Config
from utils import GCSUploadCallback, EarlyStoppingTrainingLossCallback, FreeMemoryCallback

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

    # Find the masked token positions
    mask_token_indices = torch.where(labels != -100)

    # Extract predicted token IDs at masked positions
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    preds = preds[mask_token_indices]

    # Extract true token IDs at masked positions
    true_labels = torch.tensor(labels)[mask_token_indices]

    # Convert to numpy for sklearn metrics
    preds = preds.cpu().numpy()
    true_labels = true_labels.cpu().numpy()

    return {
        "accuracy": accuracy_score(true_labels, preds),
        "f1": f1_score(true_labels, preds, average="weighted")
    }


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        device = 'cpu'
        
        if isinstance(model, torch.nn.DataParallel):
            device = model.module.device
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop("labels")

        
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        mask_token_indices = torch.where(labels != -100)
        
        # get mask_token_ids of all inputs
        logits = logits[mask_token_indices[0], mask_token_indices[1], :]
        labels = labels[mask_token_indices[0], mask_token_indices[1]]
        
        print(f'logits.shape: {logits.shape}')
        print(f'labels.shape: {labels.shape}')
        loss = F.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def train(bkt_upload=True,num_epochs=6,
          batch_size=8,
          grad_accum=4,
          save_steps=300,
        ):
    
    tokenized_datasets = tokenize()

    training_args = TrainingArguments(
        run_name=Config.PROJECT_NAME,
        output_dir=Config.OUTPUT_DIR,  # output directory
        num_train_epochs=num_epochs,  # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
        learning_rate=5e-5,
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        report_to="wandb",
        bf16=True,
    )

    es_callback = EarlyStoppingTrainingLossCallback(patience=3)
    gcs_callback = GCSUploadCallback(bkt_upload=bkt_upload)
    fm_callback = FreeMemoryCallback()
    trainer = Trainer(
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
    parser.add_argument("-epochs", "--epochs", type=int, help="Number of epochs", default=1)
    parser.add_argument("-batch_size", "--batch_size", type=int, help="Number of train batches", default=4)
    # parser.add_argument("-eval_batch_size", "--eval_batch_size", type=int, help="Number of eval batches", default=64)
    parser.add_argument("-accum_steps", "--accum_steps", type=int, help="grad accumulation steps", default=2)
    parser.add_argument("-save_steps", "--save_steps", type=int, help="save steps", default=10)
    # parser.add_argument("-eval_steps", "--eval_steps", type=int, help="eval steps", default=10)
    
    
    parser.add_argument("-u", "--upload", action="store_true", help="Enable uploads")
    args = parser.parse_args()
    
    bkt_upload = False
    if args.upload:
        print(f'bucket upload enabled')
        bkt_upload=True
    else:
        print(f'bucket upload disabled')
    
    
    train(bkt_upload=bkt_upload, 
          num_epochs=args.epochs,
          batch_size=args.batch_size,
          grad_accum=args.accum_steps,
          save_steps=args.save_steps,
        )


