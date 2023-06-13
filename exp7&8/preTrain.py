seed = 3407

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=str)
parser.add_argument('-t', type=str)

args = parser.parse_args()

if args.n == 0:
    model_name = "bert-base-chinese"
    model_path = 'bert-base-chinese'
elif args.n == 1:
    model_name = "chinese-bert-wwm"
    model_path = 'hfl/chinese-bert-wwm'
elif args.n == 2:
    model_name = "chinese-bert-wwm-ext"
    model_path = 'hfl/chinese-bert-wwm-ext'
elif args.n == 3:
    model_name = "chinese-roberta-wwm-ext"
    model_path = 'hfl/chinese-roberta-wwm-ext'

data_disk_path = "../dataset/preTrain"
mlm_probability = 0.15
save_steps = 25000
learning_rate = 2e-5
lr_scheduler_type = "constant"

wwm = 'wwm' in model_name

vocab_path = model_path

if args.t == 'long':
    mlm = True
    nsp = False
    long = True
    short = False
elif args.t == 'short':
    mlm = True
    nsp = False
    long = False
    short = True
elif args.t == 'nsp':
    mlm = False
    nsp = True
    long = False
    short = False

if mlm:
    assert long != short
assert mlm != nsp

output_dir = './' + model_name + '_preTrain_' + args.t
logging_dir = './' + model_name + '_log_' + args.t
model_save_path = './' + model_name + '_csl_' + args.t
train_dataset = data_disk_path + "/dataset_" + args.t

if long:
    batch_size = 32
    num_train_epochs = 100
if short:
    batch_size = 32
    # batch_size = 192
    num_train_epochs = 100
if nsp:
    batch_size = 32
    num_train_epochs = 100

if mlm:
    max_steps = save_steps * 12 + 1
if nsp:
    max_steps = save_steps * 4 + 1

import random

import numpy as np
import torch
from transformers import (BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction,
                          DataCollatorForLanguageModeling, DataCollatorForWholeWordMask, DataCollatorWithPadding,
                          TrainingArguments, Trainer)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


seed_everything(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained(vocab_path)

from datasets import load_from_disk

if mlm and not nsp:
    train_dataset = data_disk_path + "/dataset_" + args.t

if nsp and not mlm:
    train_dataset = data_disk_path + "/dataset_paragraphs"

dataset = load_from_disk(train_dataset)

if mlm and not nsp:
    model = BertForMaskedLM.from_pretrained(model_path).to(device)
    if wwm:
        if long:
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "chinese_ref"])
        elif short:
            dataset.set_format(type="torch", columns=["input_ids", "chinese_ref"])
        data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
    else:
        dataset = dataset.remove_columns("chinese_ref")
        if long:
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        elif short:
            dataset.set_format(type="torch", columns=["input_ids"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

if nsp and not mlm:
    model = BertForNextSentencePrediction.from_pretrained(model_path).to(device)
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    data_collator = DataCollatorWithPadding(tokenizer)

print("model_name:", model_name)
print("mlm:", mlm)
print("wwm:", wwm)
print("long:", long)
print("short:", short)
print("nsp:", nsp)

training_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=num_train_epochs,
                                  max_steps=max_steps, learning_rate=learning_rate, lr_scheduler_type=lr_scheduler_type,
                                  per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                  evaluation_strategy='steps',  # 评价策略
                                  logging_dir=logging_dir, logging_strategy='steps', logging_steps=5000,
                                  save_strategy='steps', save_steps=save_steps, prediction_loss_only=True, seed=seed,
                                  bf16=True)

trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset['train'],
                  eval_dataset=dataset['test'])

trainer.train()  # trainer.train(resume_from_checkpoint=model_path)

# trainer.save_model(model_save_path)
# tokenizer.save_pretrained(model_save_path)
