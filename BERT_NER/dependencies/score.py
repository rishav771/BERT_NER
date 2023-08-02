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
import pandas as pd
from collections import OrderedDict


def init():
    global model
    global device

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
    
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "uc1_token")
    model = mlflow.pyfunc.load_model(model_path)
    logging.info("Init complete")


def extract_entities_from_bio(sentence, bio_tags):
    entities = {}
    current_entity = None
    current_entity_type = None
    words = sentence.split()
    tags = bio_tags.split()

    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current_entity:
                entity_value = " ".join(current_entity)
                entities[entity_value] = current_entity_type

            current_entity = [word]
            current_entity_type = tag[2:]
        elif tag.startswith("I-"):
            if current_entity:
                current_entity.append(word)
        else:
            if current_entity:
                entity_value = " ".join(current_entity)
                entities[entity_value] = current_entity_type
                current_entity = None
                current_entity_type = None

    if current_entity:
        entity_value = " ".join(current_entity)
        entities[entity_value] = current_entity_type

    return entities


def run(json_input):

    # Load the JSON input
    data = json.loads(json_input)

    # Initialize the output_data with the given structure
    output_data = {
        "status": "success",
        "result": [{
            "general": [{
                "types": ["Dog supplies"],
                "all_attributes": [{
                    "Title": "string",
                    "Description": "string",
                    "Price": "decimal",
                    "Category": "string",
                    "Type": "string",
                    "Image": "file"
                }],
                "common_attributes": [{
                    "Title": "string",
                    "Description": "string",
                    "Price": "file",
                    "Category": "string"
                }]
            }]
        }],
        "data": []
    }

    # Create the "tags" field for the "Category" key
    tags_data = [{
        "attribute": "category",
        "probability": "100%"
    }]

    # Loop through each data item and process them individually
    for item in data["data"]:
        # Create an ordered dictionary to maintain the key order for each data item
        category_output = OrderedDict()
        sample =  pd.DataFrame({ 'text':[item['Title']]})
        bio_tags = " ".join(model.predict(sample))
        # Extract entities for the "Title" attribute using the given function
        title_entities = extract_entities_from_bio(item["Title"], bio_tags)  # Assuming 'bio_tags' is defined earlier

        # Add the keys in the desired order for the current data item
        for key in ["Title", "Description", "Price", "Category", "Image"]:
            if key == "Category":
                category_output[key] = [
                    {
                        "input": item[key],
                        "status": "mapped",
                        "tags": {
                            item[key]: tags_data
                        }
                    }
                ]
            elif key == "Title":
                if title_entities:
                    title_tags = []
                    for entity_key, entity_value in title_entities.items():
                        if entity_key in item[key]:
                            title_tags.append({entity_key: [{"attribute": entity_value}]})
                    category_output[key] = [
                        {
                            "input": item[key],
                            "status": "mapped",
                            "tags": title_tags
                        }
                    ]
                else:
                    category_output[key] = [
                        {
                            "input": item[key],
                            "status": "unmapped"
                        }
                    ]
            else:
                category_output[key] = [
                    {
                        "input": item[key],
                        "status": "unmapped"
                    }
                ]

        # Add the output format for the current data item to the "data" list
        output_data["data"].append(category_output)


    return output_data
