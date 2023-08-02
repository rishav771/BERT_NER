import os,pathlib
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch 
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification,BertConfig,AdamW,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
import torch.multiprocessing as mp
import nvidia_smi
import numpy as np
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel, PythonModelContext
import mlflow
from typing import Dict
import ast
import logging


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
label_all_tokens = False
nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
# add arguments
parser.add_argument("--data_dir", type=str, help="directory containing CIFAR-10 dataset")
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
args = parser.parse_args()
logging.info("input parameters: %s", vars(args))

df = pd.read_csv('./data/output_fashion.csv', names=["text", "labels", "category"], skiprows=1)
# Drop the last column
df = df.drop(columns="category")
# Function to convert the string representation to a list
def convert_to_list(string_representation):
    return ast.literal_eval(string_representation)
    
# Apply the function to the 'bio_tag' column for all rows
df['labels'] = df['labels'].apply(convert_to_list)

labels = [i for i in df['labels'].values.tolist()]
unique_labels = set()
for lb in labels:
    [unique_labels.add(i) for i in lb if i not in unique_labels]
# Sort the unique_labels
sorted_unique_labels = sorted(unique_labels)
# Create labels_to_ids and ids_to_labels mappings
labels_to_ids = {label: idx for idx, label in enumerate(sorted_unique_labels)}
ids_to_labels = {idx: label for idx, label in enumerate(sorted_unique_labels)}

# Function to convert the space-separated string to a list
def convert_to_str(string_representation):
    return " ".join(string_representation)

# Apply the function to the 'bio_tag' column for all rows
df['labels'] = df['labels'].apply(convert_to_str)


class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(sorted_unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output


class BertTokenClassifier(PythonModel):
    def load_context(self, context: PythonModelContext):
        from transformers import BertTokenizerFast, BertForTokenClassification,BertConfig
        import os
        import pandas as pd 

        config_file = os.path.dirname(context.artifacts["config"])
        self.config = BertConfig.from_pretrained(config_file)
        self.tokenizer = BertTokenizerFast.from_pretrained(config_file)
        self.model = BertForTokenClassification.from_pretrained(config_file, config=self.config)
        if torch.cuda.is_available():
            print('[INFO] Model is being sent to CUDA device as GPU is available')
            self.model = self.model.cuda()
        else:
            print('[INFO] Model will use CPU runtime')
        
        _ = self.model.eval()
    
    def align_word_ids(self,texts):
  
        tokenized_inputs = self.tokenizer(texts, padding='max_length', max_length=120, truncation=True)

        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(1)
                except:
                    label_ids.append(-100)
            else:
                try:
                    label_ids.append(1 if self.label_all_tokens else -100)
                except:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        return label_ids
        
    def predict(self, context: PythonModelContext, data: pd.DataFrame):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            self.model = self.model.cuda()

        text = self.tokenizer(data['text'][0], padding='max_length', max_length=120, truncation=True, return_tensors="pt")

        mask = text['attention_mask'].to(device)
        input_id = text['input_ids'].to(device)
        label_ids = torch.Tensor(align_word_ids(data['text'][0])).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.model(input_id, mask, None)
        
        logits_clean = logits[0][label_ids != -100]
        predictions = logits_clean.argmax(dim=1).tolist()
        prediction_label = [ids_to_labels[i] for i in predictions]

        return prediction_label


def align_word_ids(texts):
  
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=120, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=120, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):

        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 120, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels


def evaluate(model, val_loader, device,df_val):
    total_acc_test = 0.0
    model.eval()

    with torch.no_grad():
        for test_data, test_label in val_loader:

                test_label = test_label.to(device)
                mask = test_data['attention_mask'].squeeze(1).to(device)

                input_id = test_data['input_ids'].squeeze(1).to(device)

                loss, logits = model(input_id, mask, test_label)

                for i in range(logits.shape[0]):

                    logits_clean = logits[i][test_label[i] != -100]
                    label_clean = test_label[i][test_label[i] != -100]

                    predictions = logits_clean.argmax(dim=1)
                    acc = (predictions == label_clean).float().mean()
                    total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(df_val): .3f}')


# define functions
def train_model(model, train_loader, optimizer, device,epoch_num,df_train,df_val,val_loader):
    total_acc_train = 0
    total_loss_train = 0
    model.train()

    for train_data, train_label in tqdm(train_loader):
        train_label = train_label.to(device)
        mask = train_data['attention_mask'].squeeze(1).to(device)
        input_id = train_data['input_ids'].squeeze(1).to(device)

        loss, logits = model(input_id, mask, train_label)

        for i in range(logits.shape[0]):

            logits_clean = logits[i][train_label[i] != -100]
            label_clean = train_label[i][train_label[i] != -100]

            predictions = logits_clean.argmax(dim=1)
            acc = (predictions == label_clean).float().mean()
            total_acc_train += acc
            total_loss_train += loss.item()
            
        loss.backward()
        optimizer.step()

    model.eval()

    total_acc_val = 0
    total_loss_val = 0

    for val_data, val_label in val_loader:

        val_label = val_label.to(device)
        mask = val_data['attention_mask'].squeeze(1).to(device)
        input_id = val_data['input_ids'].squeeze(1).to(device)

        loss, logits = model(input_id, mask, val_label)

        for i in range(logits.shape[0]):

            logits_clean = logits[i][val_label[i] != -100]
            label_clean = val_label[i][val_label[i] != -100]

            predictions = logits_clean.argmax(dim=1)
            acc = (predictions == label_clean).float().mean()
            total_acc_val += acc
            total_loss_val += loss.item()

    acc_t = 4*(total_acc_train / len(df_train))
    acc_v = 4*(total_acc_val / len(df_val))
    loss_t = 4*(total_loss_train / len(df_train))
    loss_v = 4*(total_loss_val / len(df_val))                    
    print(
            f'Epochs: {epoch_num + 1} | Loss: {loss_t: .3f} | Accuracy: {acc_t: .3f} | Val_Loss: {loss_v: .3f} | Accuracy: {acc_v: .3f}')

       

def main():
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Shuffle the DataFrame rows
    df_new = df.sample(frac=1, random_state=42)
    # Calculate the indices to split the DataFrame
    train_size = int(0.9 * len(df_new))
    val_size = len(df_new) - train_size
    # Split the DataFrame into df_train and df_val
    df_train = df_new[:train_size]
    df_val = df_new[train_size:]


    # B. Prepare For Distributed Training
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    is_distributed = world_size > 1
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")


    # C. Perform Certain Tasks Only In Specific Processes
    if local_rank == 0:
        train_set = DataSequence(df_train)
    if is_distributed:
        torch.distributed.barrier()
    if local_rank != 0:
        train_set = DataSequence(df_train)


    # D. Create Distributed Sampler and Data Loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if is_distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler)

    # C. Perform Certain Tasks Only In Specific Processes
    if local_rank == 0:
        val_set = DataSequence(df_val)
    if is_distributed:
        torch.distributed.barrier()
    if local_rank != 0:
        val_set = DataSequence(df_val)


    # D. Create Distributed Sampler and Data Loader
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set) if is_distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler)

  
    # E. Initialize Model Using DistributedDataParallel
    model = BertModel().to(device)
    

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)
    

    # F. Set Learning Rate and Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    
    
    # G. Update Distributed Sampler On Each Epoch
    for epoch in range(args.epochs):
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(f"|Device {i}| Mem Free: {mem.free/1024**2:5.2f}MB / {mem.total/1024**2:5.2f}MB | gpu-util: {util.gpu:3.1%} | gpu-mem: {util.memory:3.1%} |")
        print("start",epoch)

        logging.info("Epoch %d", epoch + 1)
        if is_distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        train_model(model, train_loader, optimizer, device,epoch,df_train,df_val,val_loader)

    

    if rank == 0:
        val_loader = torch.utils.data.DataLoader(DataSequence(df_val), batch_size=args.batch_size, shuffle=False)
        evaluate(model, val_loader, device,df_val)

    #         metrics = {
    #     "validation_accuracy": validation_accuracy
    # }
    # mlflow.log_metrics(metrics, step=epoch_num)
        model_path = './model'
        model.module.bert.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        artifacts = { pathlib.Path(file).stem: os.path.join(model_path, file) 
                    for file in os.listdir(model_path) 
                    if not os.path.basename(file).startswith('.') }

        with mlflow.start_run():
            mlflow.pyfunc.log_model('uc1_token', 
                                    python_model=BertTokenClassifier(), 
                                    artifacts=artifacts, 
                                    registered_model_name='uc1_token')
        
if __name__ == '__main__':
    main()