import logging
import os
import json
import mlflow
from io import StringIO
from mlflow.pyfunc.scoring_server import infer_and_parse_json_input, predictions_to_json
import sys
from time import strftime, localtime
from collections import Counter
from pytorch_transformers import BertTokenizer
import random
import numpy as np
import torch
from tqdm import tqdm
import torch

# original = torch.load


# def load(*args):
#     return torch.load(*args, map_location=torch.device("cpu"),pickle_module=None)


# def init():
#     global model
#     global device
#     global tokenizer

#     cuda_available = torch.cuda.is_available()
#     device = "cuda" if cuda_available else "cpu"

#     if cuda_available:
#         print(f"[INFO] CUDA version: {torch.version.cuda}")
#         print(f"[INFO] ID of current CUDA device: {torch.cuda.current_device()}")
#         print("[INFO] nvidia-smi output:")
#     else:
#         print(
#             "[WARN] CUDA acceleration is not available. This model takes hours to run on medium size data."
#         )

#     model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "use-case1-model")
#     # "model" is the path of the mlflow artifacts when the model was registered. For automl
#     # models, this is generally "mlflow-model".

#     model = mlflow.pyfunc.load_model(model_path, *{'map_location': torch.device('cpu')})

#     logging.info("Init complete")

def init():
    global model
    global device
    global tokenizer

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    if cuda_available:
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] ID of current CUDA device: {torch.cuda.current_device()}")
        print("[INFO] nvidia-smi output:")
    else:
        print(
            "[WARN] CUDA acceleration is not available. This model takes hours to run on medium size data."
        )

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "use-case1-model")

    model = mlflow.pytorch.load_model(model_path, map_location=torch.device('cpu'))
    logging.info("Init complete")


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

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

max_len = 40

def X_padding(ids):
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids

tag_max_len = 6

def tag_padding(ids):
    if len(ids) >= tag_max_len:
        return ids[:tag_max_len]
    ids.extend([0] * (tag_max_len - len(ids)))
    return ids

def nobert4token(tokenizer, title, attribute):
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

    title_list = tokenizer.convert_tokens_to_ids(title_list)
    attribute_list = tokenizer.convert_tokens_to_ids(attribute_list)


    return title_list, attribute_list

def run(data):

    json_data = json.loads(data) 

    title = json_data["input_data"]["title"]
    att = json_data["input_data"]["attributes"]
    
    result = {}

    for i in range(len(title)):

        my_dict = {}
        for j in range(len(att)):
            
            attr = att[i][j]

            t, a = nobert4token(tokenizer, title[i], attr)

            x = X_padding(t)
            y = tag_padding(a)

            tensor_a = torch.tensor(y, dtype=torch.int32)
            tensor_a = torch.unsqueeze(tensor_a, dim=0).to(device)

            tensor_t = torch.tensor(x, dtype=torch.int32)
            tensor_t = torch.unsqueeze(tensor_t, dim=0).to(device)

            output = model([tensor_t, tensor_a])

            predict_list = output.tolist()[0]

            for k in range(len(predict_list)):
                start_p, end_p = 0, 0
                for index, value in enumerate(predict_list[k]):
                    if value == 1:
                        start_p = index
                        ind = index
                        while predict_list[k][ind] != 3:
                            ind = ind + 1
                            end_p = ind
                preds = tensor_t[k][start_p:end_p]
                words_p = tokenizer.convert_ids_to_tokens(
                    [k.item() for k in preds.cpu() if k.item() > 0])
            
            my_dict[attr] = " ".join(words_p)

        result[title[i]] = my_dict


    return result
