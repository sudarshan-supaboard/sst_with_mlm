from transformers import BertForMaskedLM, BertTokenizer
from config import Config
from peft import LoraConfig, get_peft_model # type: ignore
from preprocess import dataset

model_path = "google-bert/bert-base-uncased"

model = BertForMaskedLM.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path,
                                          do_lower_case=True,
                                          strip_accents=True)

lora_config = LoraConfig(
    r=8, # Rank
    lora_alpha=16,
    target_modules=["query", "value", "key"],
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS",# or any other appropriate task type,
    modules_to_save=['cls.decoder']
)


# Apply LoRA using PEFT
model = get_peft_model(model, lora_config)
model = model.to(Config.device())


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
    pprint(model)
    model.print_trainable_parameters()
    
