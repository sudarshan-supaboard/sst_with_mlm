import torch
import torch.nn.functional as F
import os
import json
import evaluate
import numpy as np

from huggingface_hub import login as hf_login
from wandb import login as wandb_login
from transformers import Trainer, TrainingArguments
from model import tokenizer, model, tokenized_datasets
from dotenv import load_dotenv
from pprint import pprint
from config import Config
from utils import GCSUploadCallback, EarlyStoppingTrainingLossCallback

load_dotenv()

hf_key = os.environ["HUGGING_FACE_API_KEY"]
wandb_key = os.environ["WANDB_API_KEY"]


hf_login(token=hf_key)
wandb_login(key=wandb_key)


class CustomTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):

        device = model.module.device
        
        labels = inputs.pop("labels").to(device)

        # inputs
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)

        mask_token_id = tokenizer.mask_token_id
        outputs = model(**inputs)
        logits = outputs.logits

        # get mask_token_ids of all inputs
        mask_token_indices = torch.where(inputs["input_ids"] == mask_token_id)
        
        logits = logits[mask_token_indices[0], mask_token_indices[1], :]

        # calculate loss
        loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


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


training_args = TrainingArguments(
    output_dir=Config.OUTPUT_DIR,  # output directory
    num_train_epochs=1,  # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    # gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,  # batch size for evaluation
    warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
    learning_rate=5e-5,
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=1,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=10,
    save_steps=10,
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="wandb",
    # bf16=True
    optim="adamw_bnb_8bit",
)

es_callback = EarlyStoppingTrainingLossCallback()
gcs_callback = GCSUploadCallback()


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


def save_best_model():
    with open("best_checkpoint.json", "w") as f:
        json.dump(obj={"checkpoint": best_checkpoint}, fp=f)
    print(f"Best Checkpoint: {best_checkpoint} Saved.")


save_best_model()
