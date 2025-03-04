import torch

from transformers import BertForMaskedLM, BertTokenizer
from config import Config
from peft import LoraConfig, get_peft_model # type: ignore
from preprocess import dataset

model = BertForMaskedLM.from_pretrained(Config.MODEL_PATH, 
                                        device_map="auto")
tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH,
                                          do_lower_case=True,
                                          strip_accents=True)

lora_config = LoraConfig(
    r=8, # Rank
    lora_alpha=16,
    target_modules=["query", "value", "key"],
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS",# or any other appropriate task type,
    # modules_to_save=['cls.decoder']
)

# num_layers = 12  # BERT-base has 12 layers
# target_layers = [f"encoder.layer.{i}.attention.self" for i in range(num_layers - 8, num_layers)]

# # Modify the target modules to include only the last 8 layers
# lora_config.target_modules = [
#     f"{layer}.{module}" for layer in target_layers for module in ["query", "key", "value"]
# ]


# Apply LoRA using PEFT
model = get_peft_model(model, lora_config)
model = model.to(Config.device())

model.print_trainable_parameters()

def tokenize_function(examples):
    examples["text"] = [f"'{sentence}', emotion of the given text is [MASK]?" for sentence in examples["text"]]

    batch_inputs = tokenizer(examples["text"], padding="max_length",
                             truncation=True, return_tensors="pt")
    batch_inputs["label"] = tokenizer.convert_tokens_to_ids(examples["label"])
    return batch_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets.set_format(type='torch',
                              columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
                              device='cpu',
                              output_all_columns=True)

if __name__ == '__main__':
    from pprint import pprint
    # pprint(model)
    # model.print_trainable_parameters()
    