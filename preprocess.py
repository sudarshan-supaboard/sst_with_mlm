import kagglehub
import pandas as pd

from datasets import Dataset, DatasetDict
from pathlib import Path
from pprint import pprint


# Download latest version
path = kagglehub.dataset_download("debarshichanda/goemotions")

print("Path to dataset files:", path)

ds_path = Path(path)
emotions = ds_path / 'data/emotions.txt'
train_path = ds_path / 'data/train.tsv'
test_path = ds_path / 'data/test.tsv'
valid_path = ds_path / 'data/dev.tsv'

train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")
valid_df = pd.read_csv(valid_path, sep="\t")

# shuffle the train_df
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac = 1).reset_index(drop=True)
valid_df = valid_df.sample(frac = 1).reset_index(drop=True)

with open(emotions) as f:
    idx_to_labels = f.readlines()

idx_to_labels = list(map(lambda x: x.strip(), idx_to_labels))

# create labels_to_idx
labels_to_idx = {}
for i, label in enumerate(idx_to_labels):
    labels_to_idx[label] = i


train_df.columns = ["text", "label", "unnamed"]
test_df.columns = ["text", "label", "unnamed"]
valid_df.columns = ["text", "label", "unnamed"]

train_df.drop(columns=['unnamed'], inplace=True)
test_df.drop(columns=['unnamed'], inplace=True)
valid_df.drop(columns=['unnamed'], inplace=True)


def map_emotions(x):
   return idx_to_labels[int(x.split(',')[0])]

train_df["label"] = train_df["label"].apply(map_emotions)
test_df["label"] = test_df["label"].apply(map_emotions)
valid_df["label"] = valid_df["label"].apply(map_emotions)

pprint({
    'train_df.shape': train_df.shape,
    'test_df.shape': test_df.shape,
    'valid_df.shape': valid_df.shape,
})

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({"train": train_dataset, "valid": valid_dataset, "test": test_dataset})
