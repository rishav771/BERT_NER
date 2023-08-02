# Import modules
from pytorch_transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import pickle
import random
import numpy as np
from collections import Counter
import os
import argparse
import logging
from unittest import mock
import mlflow


# Function to check if the character is English or not
def is_english_char(cp):
    """Checks whether CP is the codepoint of an English character."""
    if (
        (cp >= 0x0041 and cp <= 0x005A)
        or (cp >= 0x0061 and cp <= 0x007A)  # uppercase A-Z
        or (cp >= 0x00C0 and cp <= 0x00FF)  # lowercase a-z
        or (cp >= 0x0100 and cp <= 0x017F)  # Latin-1 Supplement
        or (cp >= 0x0180 and cp <= 0x024F)  # Latin Extended-A
        or (cp >= 0x1E00 and cp <= 0x1EFF)  # Latin Extended-B
        or (cp >= 0x2C60 and cp <= 0x2C7F)  # Latin Extended Additional
        or (cp >= 0xA720 and cp <= 0xA7FF)  # Latin Extended-C
        or (cp >= 0xAB30 and cp <= 0xAB6F)  # Latin Extended-D
        or (cp >= 0xFB00 and cp <= 0xFB06)  # Latin Extended-E
    ):  # Alphabetic Presentation Forms
        return True

    return False


# Function to convert the text to word embeddings using BERT
def nobert4token(tokenizer, title, attribute, value):
    def get_char(sent):
        tmp = []
        s = ""
        for char in sent.strip():
            if char.strip():
                cp = ord(char)
                if is_english_char(cp):
                    if s:
                        tmp.append(s)
                    tmp.append(char)
                    s = ""
                else:
                    s += char
            elif s:
                tmp.append(s)
                s = ""
        if s:
            tmp.append(s)
        return tmp

    title_list = get_char(title)
    attribute_list = get_char(attribute)
    value_list = get_char(value)

    tag_list = ["O"] * len(title_list)
    for i in range(0, len(title_list) - len(value_list)):
        if title_list[i : i + len(value_list)] == value_list:
            for j in range(len(value_list)):
                if j == 0:
                    tag_list[i + j] = "B"
                else:
                    tag_list[i + j] = "I"

    title_list = tokenizer.convert_tokens_to_ids(title_list)
    attribute_list = tokenizer.convert_tokens_to_ids(attribute_list)
    value_list = tokenizer.convert_tokens_to_ids(value_list)
    tag_list = [TAGS[i] for i in tag_list]

    return title_list, attribute_list, value_list, tag_list

#Add paddinng to title to make it of same length
def X_padding(ids):
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids


tag_max_len = 20

#Add paddinng to tags to make it of same length
def tag_padding(ids):
    if len(ids) >= tag_max_len:
        return ids[:tag_max_len]
    ids.extend([0] * (tag_max_len - len(ids)))
    return ids

#Add paddinng to attribute to make it of same length
def val_padding(ids):
    if len(ids) >= val_max_len:
        return ids[:val_max_len]
    ids.extend([0] * (val_max_len - len(ids)))
    return ids

#Gebrate BIO tag and word embedding
def rawdata2pkl4nobert(path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    titles = []
    attributes = []
    values = []
    tags = []
    # for index in range(length):
    #     if brand[index] in title[index] and is_english_char(ord(brand[index][0])):
    #         t, a, v, tag = nobert4token(tokenizer, title[index], "brand", brand[index])
    #         titles.append(t)
    #         attributes.append(a)
    #         values.append(v)
    #         tags.append(tag)

    with open(path, 'r',encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

        for index, line in enumerate(tqdm(lines[:50000])):
            title, attribute, value = line.split('$$$')
            title, attribute, value, tag = nobert4token(tokenizer, title, attribute, value)
            titles.append(title)
            attributes.append(attribute)
            values.append(value)
            tags.append(tag)


    print([tokenizer.convert_ids_to_tokens(i) for i in titles[:1]])
    print([[id2tags[j] for j in i] for i in tags[:1]])
    print([tokenizer.convert_ids_to_tokens(i) for i in attributes[:1]])
    print([tokenizer.convert_ids_to_tokens(i) for i in values[:1]])

    df = pd.DataFrame(
        {"titles": titles, "attributes": attributes, "values": values, "tags": tags},
        index=range(len(titles)),
    )
    df["x"] = df["titles"].apply(X_padding)
    df["y"] = df["tags"].apply(X_padding)
    df["att"] = df["attributes"].apply(tag_padding)
    df["val"] = df["values"].apply(val_padding)

    index = list(range(len(titles)))
    random.shuffle(index)
    train_index = index[: int(0.9 * len(index))]
    valid_index = index[int(0.9 * len(index)) : int(0.96 * len(index))]
    test_index = index[int(0.96 * len(index)) :]

    train = df.loc[train_index, :]
    valid = df.loc[valid_index, :]
    test = df.loc[test_index, :]

    train_x = np.asarray(list(train["x"].values))
    train_att = np.asarray(list(train["att"].values))
    train_y = np.asarray(list(train["y"].values))

    valid_x = np.asarray(list(valid["x"].values))
    valid_att = np.asarray(list(valid["att"].values))
    valid_y = np.asarray(list(valid["y"].values))

    test_x = np.asarray(list(test["x"].values))
    test_att = np.asarray(list(test["att"].values))
    test_value = np.asarray(list(test["val"].values))
    test_y = np.asarray(list(test["y"].values))

    print(test_x[0:1])
    print(test_att[0:1])
    print(test_value[0:1])
    print(test_y[0:1])

    with open(os.path.join(args.train_data, "new_container.pkl"), "wb") as outp:
        pickle.dump(train_x, outp)
        pickle.dump(train_att, outp)
        pickle.dump(train_y, outp)
        pickle.dump(valid_x, outp)
        pickle.dump(valid_att, outp)
        pickle.dump(valid_y, outp)
        pickle.dump(test_x, outp)
        pickle.dump(test_att, outp)
        pickle.dump(test_value, outp)
        pickle.dump(test_y, outp)


if __name__ == "__main__":
    TAGS = {"": 0, "B": 1, "I": 2, "O": 3}
    id2tags = {v: k for k, v in TAGS.items()}

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--train_data", type=str, help="path to train data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    # # Read the Excel sheet into a pandas dataframe
    # df = pd.read_csv(args.data)

    # df1 = df.dropna(subset=["Brand"])

    # print(df1["Product Title"][0:5])

    # title = df1["Product Title"].tolist()
    # brand = df1["Brand"].tolist()

    # max_len = len(max(title, key=len))
    # val_max_len = len(max(brand, key=len))

    max_len = 100
    val_max_len = 50

    rawdata2pkl4nobert(args.data)

    # Stop Logging
    mlflow.end_run()
