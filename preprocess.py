import kagglehub
import pandas as pd

from datasets import Dataset, DatasetDict
from pathlib import Path
from pprint import pprint

from config import Config
# Download latest version
path = kagglehub.dataset_download(Config.DATASET_PATH)

print("Path to dataset files:", path)

ds_path = Path(path)
emotions = ds_path / 'emotions.txt'
train_path = ds_path / 'train.csv'
test_path = ds_path / 'test.csv'
valid_path = ds_path / 'val.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
valid_df = pd.read_csv(valid_path)

# shuffle the train_df
train_df = train_df.sample(frac=1, random_state=Config.RANDOM_STATE, ignore_index=True)
test_df = test_df.sample(frac = 1, random_state=Config.RANDOM_STATE, ignore_index=True)
valid_df = valid_df.sample(frac = 1, random_state=Config.RANDOM_STATE, ignore_index=True)

with open(emotions) as f:
    idx_to_labels = f.readlines()

idx_to_labels = list(map(lambda x: x.strip(), idx_to_labels))

# create labels_to_idx
labels_to_idx = {}
for i, label in enumerate(idx_to_labels):
    labels_to_idx[label] = i


pprint({
    'train_df.shape': train_df.shape,
    'test_df.shape': test_df.shape,
    'valid_df.shape': valid_df.shape,
})

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({"train": train_dataset, "valid": valid_dataset, "test": test_dataset})

if __name__ == '__main__':
    print(dataset['train'][15])