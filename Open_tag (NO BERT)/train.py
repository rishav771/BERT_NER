import os
import sys
from time import strftime, localtime
from collections import Counter
from pytorch_transformers import BertTokenizer
import random
import numpy as np
import torch
import models
from utils import get_dataloader
from seqeval.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import mlflow
import argparse
import pandas as pd
import warnings
import logging
from mlflow.models.signature import infer_signature

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# input and output arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, help="path to train data")
parser.add_argument("--data", type=str, help="path to input data")
parser.add_argument("--model", type=str, help="path to model file")
args = parser.parse_args()


def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


class DefaultConfig(object):
    env = "default"
    vis_port = 8097
    model = "OpenTag2019"
    pretrained_bert_name = "bert-base-cased"

    pickle_path = select_first_file(args.train_data)
    load_model_path = None

    batch_size = 32  # batch size
    embedding_dim = 768
    hidden_dim = 1024
    tagset_size = 4
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    max_epoch = 1
    lr = 2e-5  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5
    dropout = 0.2
    seed = 1234
    device = "cuda"

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = torch.device("cuda") if opt.use_gpu else torch.device("cpu")


opt = DefaultConfig()

# Start Logging
mlflow.start_run()

# enable autologging
mlflow.pytorch.autolog()

print("version:", torch.__version__)

os.makedirs("./outputs", exist_ok=True)

def get_attributes(path=args.data):
    atts = []
    with open(path, 'r',encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
        for line in lines[:50000]:
            title, attribute, value = line.split('$$$')
            atts.append(attribute)
    return [item[0] for item in Counter(atts).most_common(10)]


def main(**kwargs):
    torch.cuda.empty_cache()
    log_file = "{}-{}.log".format(opt.model, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    att_list = get_attributes()

    tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
    tags2id = {"": 0, "B": 1, "I": 2, "O": 3}
    id2tags = {v: k for k, v in tags2id.items()}

    opt._parse(kwargs)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # step1: configure model
    model = getattr(models, opt.model)(opt)
    if opt.load_model_path:
        the_model = torch.load(PATH)
    model.to(opt.device)

    # step2: data
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(opt)

    # step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4 train
    for epoch in range(opt.max_epoch):
        model.train()
        for ii, batch in tqdm(enumerate(train_dataloader)):
            # train model
            optimizer.zero_grad()
            x = batch["x"].to(opt.device)
            y = batch["y"].to(opt.device)
            att = batch["att"].to(opt.device)
            inputs = [x, att, y]
            loss = model.log_likelihood(inputs)
            loss.backward()
            # CRF
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3)
            optimizer.step()
            if ii % opt.print_freq == 0:
                print("epoch:%04d,------------loss:%f" % (epoch, loss.item()))

    preds, labels, inp, out = [], [], [], []

    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(valid_dataloader):
            x = batch["x"].to(opt.device)
            y = batch["y"].to(opt.device)
            att = batch["att"].to(opt.device)
            inputs = [x, att]
            predict = model(inputs)
            predict_list = predict.tolist()[0]

            leng = []
            for i in y.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(tmp)

            for index, i in enumerate(predict_list):
                preds.append(
                    [id2tags[k] if k > 0 else id2tags[3] for k in i[: len(leng[index])]]
                )

            for index, i in enumerate(y.tolist()):
                labels.append(
                    [id2tags[k] if k > 0 else id2tags[3] for k in i[: len(leng[index])]]
                )

        report = classification_report(labels, preds)
        print(report)
        logger.info(report)

    data = {
        "Title": [
            " Lee posh Lactic Acid 60% Anti ageing Pigmentation Removing Glow Peel "
        ],
        "Attributes": ["Brand"],
        "Values": ["Lee posh"],
    }
    # Create a DataFrame using the dictionary
    df = pd.DataFrame(data)

    # Registering the model to the workspace
    mlflow.pytorch.log_model(
        pytorch_model=model,
        registered_model_name="use-case1-model",
        artifact_path="use-case1-model",
        input_example=df[["Title", "Attributes"]],
        conda_env=os.path.join("./dependencies", "conda.yaml"),
        code_paths=[os.path.join("./models")],
    )

    # Saving the model to a file
    mlflow.pytorch.save_model(
        pytorch_model=model,
        conda_env=os.path.join("./dependencies", "conda.yaml"),
        input_example=df[["Title", "Attributes"]],
        path=os.path.join(args.model, "use-case1-model"),
    )

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
