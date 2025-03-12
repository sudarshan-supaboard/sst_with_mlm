import torch
import torch.nn.functional as F
import os
import json
import wandb
import argparse

from huggingface_hub import login as hf_login
from wandb import login as wandb_login
from transformers import Trainer, TrainingArguments
from model import model, tokenize
from dotenv import load_dotenv
from pprint import pprint
from config import Config
from utils import GCSUploadCallback, EarlyStoppingTrainingLossCallback, Accuracy

load_dotenv()

hf_key = os.environ["HUGGING_FACE_API_KEY"]
wandb_key = os.environ["WANDB_API_KEY"]


hf_login(token=hf_key)
wandb_login(key=wandb_key)

wandb.init(project=Config.PROJECT_NAME)
# Load multiple metrics
accuracy = Accuracy()

def compute_metrics(eval_pred, compute_result: bool):
    global accuracy
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    
    logits = F.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)
    
    accuracy.add(preds = preds, labels=labels)
    
    if compute_result:
        out = {
                "accuracy": accuracy.compute()
            }
        accuracy.reset()
        return out
        


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ): 
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            print("DistributedDataParallel")
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        mask_token_indices = torch.where(labels != -100)
        
        # get mask_token_ids of all inputs
        logits = logits[mask_token_indices[0], mask_token_indices[1], :]
        labels = labels[mask_token_indices[0], mask_token_indices[1]]
               
        loss = F.cross_entropy(logits, labels)

        return (loss, logits) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        # inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        labels = inputs.get('labels')
        
        # Extract logits (model predictions)
        logits = outputs.logits
        mask_token_indices = torch.where(labels != -100)
        
        # get mask_token_ids of all inputs
        logits = logits[mask_token_indices[0], mask_token_indices[1], :]
        labels = labels[mask_token_indices[0], mask_token_indices[1]] # type: ignore
        
        loss = F.cross_entropy(logits, labels)

        if prediction_loss_only:
            return (loss)
        
        return (loss, logits, labels)


def train(bkt_upload=True,
          num_epochs=6,
          batch_size=8,
          grad_accum=4,
          save_steps=300,
          eval_steps=300,
          log_steps=10,
          eval_batch=4,
        ):
    
    tokenized_datasets = tokenize()

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,  # output directory
        num_train_epochs=num_epochs,  # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=eval_batch,
        warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
        learning_rate=5e-5,
        weight_decay=0.1,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=log_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=4,
        report_to="wandb",
        bf16=True,
        remove_unused_columns=False,
        batch_eval_metrics=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    es_callback = EarlyStoppingTrainingLossCallback(patience=3)
    gcs_callback = GCSUploadCallback(bkt_upload=bkt_upload)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics, # type: ignore
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
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=3)
    parser.add_argument("-bs", "--batch_size", type=int, help="Number of train batches", default=8)
    parser.add_argument("-as", "--accum_steps", type=int, help="grad accumulation steps", default=4)
    parser.add_argument("-ss", "--save_steps", type=int, help="save steps", default=300)
    parser.add_argument("-es", "--eval_steps", type=int, help="evaluation steps", default=300)
    parser.add_argument("-ls", "--log_steps", type=int, help="logging steps", default=10)
    parser.add_argument("-eb", "--eval_batch", type=int, help="evaluation batch size", default=32)
    
    
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
          eval_steps=args.eval_steps,
          log_steps=args.log_steps,
          eval_batch=args.eval_batch
        )


