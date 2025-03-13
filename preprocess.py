import kagglehub
import pandas as pd

from datasets import Dataset, DatasetDict
from pathlib import Path
from pprint import pprint
from config import Config
from string import Template

# Download latest version


def make_dataset():
    path = kagglehub.dataset_download(Config.DATASET_PATH)

    print("Path to dataset files:", path)

    ds_path = Path(path)
    emotions = ds_path / "emotions.txt"
    train_path = ds_path / "train.csv"
    test_path = ds_path / "test.csv"
    valid_path = ds_path / "val.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    valid_df = pd.read_csv(valid_path)

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

    new_train_df = []
    new_test_df = []
    new_valid_df = []

    for i in range(len(train_df)):
        for template in templates:
            new_train_df.append(
                {
                    "text": template.substitute(
                        {"input": train_df["text"][i], "mask": Config.MASK_TOKEN}
                    ),
                    "label": train_df["label"][i],
                }
            )

    for i in range(len(test_df)):
        for template in templates:
            new_test_df.append(
                {
                    "text": template.substitute(
                        {"input": test_df["text"][i], "mask": Config.MASK_TOKEN}
                    ),
                    "label": test_df["label"][i],
                }
            )

    for i in range(len(valid_df)):
        for template in templates:
            new_valid_df.append(
                {
                     "text": template.substitute(
                        {"input": valid_df["text"][i], "mask": Config.MASK_TOKEN}
                    ),
                    "label": valid_df["label"][i],
                }
            )

    train_df = pd.DataFrame(new_train_df)
    test_df = pd.DataFrame(new_test_df)
    valid_df = pd.DataFrame(new_valid_df)

    # shuffle the train_df
    train_df = train_df.sample(
        frac=1, random_state=Config.RANDOM_STATE, ignore_index=True
    )
    test_df = test_df.sample(
        frac=1, random_state=Config.RANDOM_STATE, ignore_index=True
    )
    valid_df = valid_df.sample(
        frac=1, random_state=Config.RANDOM_STATE, ignore_index=True
    )

    with open(emotions) as f:
        idx_to_labels = f.readlines()

    idx_to_labels = list(map(lambda x: x.strip(), idx_to_labels))

    # create labels_to_idx
    labels_to_idx = {}
    for i, label in enumerate(idx_to_labels):
        labels_to_idx[label] = i

    pprint(
        {
            "train_df.shape": train_df.shape,
            "test_df.shape": test_df.shape,
            "valid_df.shape": valid_df.shape,
        }
    )

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict(
        {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
    )

    return dataset


if __name__ == "__main__":
    dataset = make_dataset()
    print(dataset["train"][0])
