import torch

from transformers import BertForMaskedLM, BertTokenizer
from config import Config
from peft import LoraConfig, get_peft_model, TaskType # type: ignore
from preprocess import dataset

model = BertForMaskedLM.from_pretrained(Config.MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH,
                                          do_lower_case=True,
                                          strip_accents=True)

lora_config = LoraConfig(
    r=8, # Rank
    lora_alpha=16,
    target_modules=["query", "value", "key"],
    lora_dropout=0.1,
    bias="none",
)

# num_layers = 12  # BERT-base has 12 layers
# target_layers = [f"encoder.layer.{i}.attention.self" for i in range(num_layers - 8, num_layers)]

# # Modify the target modules to include only the last 8 layers
# lora_config.target_modules = [
#     f"{layer}.{module}" for layer in target_layers for module in ["query", "key", "value"]
# ]


# Apply LoRA using PEFT
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def tokenize_function(examples):
    examples["text"] = [f"'{sentence}', emotion of the given text is [MASK]?" for sentence in examples["text"]]

    batch_inputs = tokenizer(examples["text"], padding="max_length",
                             truncation=True, return_tensors="pt")
     # Convert labels (words) to token IDs
    label_token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(label) for label in examples["label"]], dtype=torch.long)

    # Create labels tensor with -100 (ignored in loss)
    labels = torch.full_like(batch_inputs["input_ids"], fill_value=-100)

    # Find the index of the [MASK] token in each sequence
    mask_token_indices = torch.where(batch_inputs["input_ids"] == tokenizer.mask_token_id)

    # Place the correct label token ID at the [MASK] position
    labels[mask_token_indices] = label_token_ids

    # Convert to Python dictionary format for Hugging Face Dataset
    return {"input_ids": batch_inputs["input_ids"].tolist(), 
            "attention_mask": batch_inputs["attention_mask"].tolist(), 
            "labels": labels.tolist()}

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets.set_format(type='torch',
                              columns=['input_ids', 'attention_mask', 'labels'],
                              device='cpu',
                              output_all_columns=False)

if __name__ == '__main__':
    from pprint import pprint
    # pprint(model)
    # model.print_trainable_parameters()
    pprint(tokenized_datasets['train'][:10])
    