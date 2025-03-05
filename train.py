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
from utils import GCSUploadCallback, EarlyStoppingTrainingLossCallback, get_memory_usage, clear_cache

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
    predictions = np.argmax(logits, axis=1)

    
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],  # type: ignore
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],  # type: ignore
    }


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        mask_token_id = tokenizer.mask_token_id
        outputs = model(**inputs)
        logits = outputs.logits

        # get mask_token_ids of all inputs
        mask_token_indices = torch.where(inputs["input_ids"] == mask_token_id)
        logits = logits[mask_token_indices[0], mask_token_indices[1], :]
        
        loss = F.cross_entropy(logits, labels)
        
        metrics = compute_metrics((logits.cpu().detach().numpy(), labels.cpu().detach.numpy()))
        metrics['train_accuracy'] = metrics.pop('accuracy')
        metrics['train_f1'] = metrics.pop('f1')
        
        self.log(metrics)

        # clear_cache()
        return (loss, outputs) if return_outputs else loss


def train(bkt_upload=True,num_epochs=1,
          batch_size=4,
          grad_accum=2, 
          eval_batch_size=64, 
          save_steps=10,
          eval_steps=10
        ):

    training_args = TrainingArguments(
        run_name=Config.PROJECT_NAME,
        output_dir=Config.OUTPUT_DIR,  # output directory
        num_train_epochs=num_epochs,  # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
        warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
        learning_rate=5e-5,
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=1,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        bf16=True
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
    parser.add_argument("-epochs", "--epochs", type=int, help="Number of epochs", default=1)
    parser.add_argument("-batch_size", "--batch_size", type=int, help="Number of train batches", default=4)
    parser.add_argument("-eval_batch_size", "--eval_batch_size", type=int, help="Number of eval batches", default=64)
    parser.add_argument("-accum_steps", "--accum_steps", type=int, help="grad accumulation steps", default=2)
    parser.add_argument("-save_steps", "--save_steps", type=int, help="save steps", default=10)
    parser.add_argument("-eval_steps", "--eval_steps", type=int, help="eval steps", default=10)
    
    
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
          eval_batch_size=args.eval_batch_size,
          grad_accum=args.accum_steps,
          save_steps=args.save_steps,
          eval_steps=args.eval_steps
        )


