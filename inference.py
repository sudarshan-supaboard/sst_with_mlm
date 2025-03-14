import torch
import pandas as pd
import re
import emoji
import random
    
from typing import List
from config import Config
from transformers import BertForMaskedLM, BertTokenizer
from transformers import RobertaForMaskedLM, RobertaTokenizer
from peft import peft_model
from pprint import pprint
from string import Template
from sklearn.metrics import accuracy_score

def get_model(model_name: str, checkpoint_name: str):
    
    model = None
    tokenizer = None
    if model_name == "bert":
        Config.set_bert_model()
        model = BertForMaskedLM.from_pretrained(Config.MODEL_PATH)
        tokenizer = BertTokenizer.from_pretrained(
            Config.MODEL_PATH, do_lower_case=True, strip_accents=True)
        checkpoint_uri = f"./{Config.OUTPUT_DIR}/{checkpoint_name}"
        model = peft_model.PeftModelForSequenceClassification.from_pretrained(
            model, checkpoint_uri
        )
        model.eval()

    elif model_name == "roberta":
        Config.set_roberta_model()
        model = RobertaForMaskedLM.from_pretrained(Config.MODEL_PATH)
        tokenizer = RobertaTokenizer.from_pretrained(Config.MODEL_PATH, do_lower_case=True, strip_accents=True)
        checkpoint_uri = f"./{Config.OUTPUT_DIR}/{checkpoint_name}"
        model = peft_model.PeftModelForSequenceClassification.from_pretrained(
            model, checkpoint_uri
        )
        model.eval()
    else:
        raise ValueError("Invalid model name")

    return model, tokenizer



def predict(model, tokenizer, inputs: List[str]):
    
    # tokenize the text
    input_ids = tokenizer(
        inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=512
    ) # type: ignore

    # get the logits
    with torch.no_grad():
        logits = model(
            input_ids=input_ids["input_ids"], attention_mask=input_ids["attention_mask"]
        ).logits # type: ignore

    # extract masked_token_id
    masked_token_id = tokenizer.mask_token_id # type: ignore
    
    masked_token_indices =  torch.where(input_ids["input_ids"] == masked_token_id)
    masked_token_logits = logits[masked_token_indices[0], masked_token_indices[1],:]
    
    # print(masked_token_logits.shape)
    
    # print(masked_token_logits.shape)
    # get top 5 words
    top_tokens = torch.topk(masked_token_logits, 5, dim=1).indices # type: ignore
    top_words = [tokenizer.decode([token]) for token in top_tokens[:, 0]] # type: ignore

    # print(top_tokens.shape)
    return top_words

def preprocess_text(text: str):
    # replace hyperlinks with space
    text = re.sub(r"http\S+", " ", text)
    
    # replace numbers with space
    text = re.sub(r"\d+", " ", text)
    
    # remove emojis using emoji library
    text = emoji.get_emoji_regexp().sub(u'', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # remove special characters
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    
    # remove extra spaces
    text = re.sub(r"\s+", " ", text)
    
    # remove leading and trailing spaces
    text = text.strip()
    
    # replace newlines, tabs and carriage returns with space
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    return text

if __name__ == "__main__":
    random.seed(42)
    
    model, tokenizer = get_model("bert", "checkpoint-2100")
    
    df = pd.read_csv("./toy_store_test.csv")
    # df = df.iloc[:100]

    preds = []
    labels = []

    templates = [
        Template(
            "You predict emotion of the given text. Don't predict stopwords, special characters and punctuation. The emotion of the text '$input' is $mask?."
        ),
        Template(
            "You predict emotion of the given text. Don't predict stopwords, special characters and punctuation. Given the text '$input', predict the emotion contained in it $mask?."
        ),
        Template(
            "You predict emotion of the given text. Don't predict stopwords, special characters and punctuation. '$input', the emotion contained in the text is $mask?."
        ),
        Template(
            "You predict sentiment of the given text. Don't predict stopwords, special characters and punctuation. '$input', the sentiment of the given text is $mask?."
        ),
        Template(
            "You predict sentiment of the given text. Don't predict stopwords, special characters and punctuation. The sentiment of the text '$input' is $mask?."
        ),
        Template(
            "You predict sentiment of the given text. Don't predict stopwords, special characters and punctuation. Given the text '$input', predict the sentiment contained in it $mask?."
        ),
    ]
    
    for i in range(0, len(df["review"]), 8):
        print(f"predicting from {i}-{i+8}")
        inputs = []
        template_idx = random.randint(0, len(templates)-1)
        
        for sent in df["review"].iloc[i : i + 8]:
            sent = preprocess_text(sent)
            a = templates[template_idx].substitute({"input": sent, "mask": Config.MASK_TOKEN})
            inputs.append(a)

        current_labels = df["rating"].iloc[i : i + 8].to_list()
        current_preds = predict(model, tokenizer, inputs)
        
        
        if len(current_preds) == len(current_labels):
            preds.extend(current_preds)
            labels.extend(current_labels)

        # print(f'inputs: {len(inputs)}, preds: {len(preds)}, labels: {len(labels)}, current_labels: {len(current_labels)}, current_preds: {len(current_preds)}')
        
        if i % 64 == 0:
            (pd.crosstab(labels, preds, normalize="index") * 100).to_csv(
                "./confmat_toy.csv"
            )

    (pd.crosstab(labels, preds, normalize="index") * 100).to_csv("./confmat_toy.csv")
