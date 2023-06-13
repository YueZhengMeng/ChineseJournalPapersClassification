max_length = 512
seed = 3407
batch_size = 32
num_labels = 13
learning_rate = 2e-5
epoch_num = 4

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=str)
parser.add_argument('-t', type=str)
parser.add_argument('-i', type=int)
args = parser.parse_args()

if args.n == 0:
    model_name = "bert-base-chinese"
elif args.n == 1:
    model_name = "chinese-bert-wwm"
elif args.n == 2:
    model_name = "chinese-bert-wwm-ext"
elif args.n == 3:
    model_name = "chinese-roberta-wwm-ext"

data_disk_path = "../dataset/preTrain"
dataset_path = data_disk_path + "/dataset_" + str(max_length) + "_head_only"
model_path = data_disk_path + '/' + model_name + '_preTrain_' + args.t + "/checkpoint-" + str(25000 * args.i)
vocab_path = 'hfl/chinese-bert-wwm'
classifier_method = "default"

print(model_path)
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


seed_everything(seed)

from datasets import load_from_disk

train_dataset = load_from_disk(dataset_path)['train']

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)

from accelerate import Accelerator

# accelerator = Accelerator()
# accelerator = Accelerator(mixed_precision='fp16')
accelerator = Accelerator(mixed_precision='bf16')
device = accelerator.device

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).to(device)

from transformers import AdamW


def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    for layer in range(12, -1, -1):
        layer_params = {'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n], 'lr': lr}
        parameters.append(layer_params)
        lr *= multiplier
    classifier_params = {
        'params': [p for n, p in model.named_parameters() if 'layer_norm' in n or 'linear' in n or 'pooling' in n],
        'lr': classifier_lr}
    parameters.append(classifier_params)
    return parameters


parameters = get_parameters(model, 2e-5, 0.95, 1e-4)
optimizer = AdamW(parameters, lr=learning_rate)

from transformers import get_linear_schedule_with_warmup

# 学习率调节器
steps_every_epoch = len(train_dataloader)
total_train_steps = steps_every_epoch * epoch_num

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_train_steps)

model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)

from tqdm import tqdm

loss_list = []
total_steps = 0


# 训练函数
def train():
    global steps_every_epoch, loss_list, total_steps
    model.train()
    epoch_train_loss = 0
    for batch in tqdm(train_dataloader):
        # 正向传播
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # 低版本transformers库在accelerate库开启混合精度训练后，模型输出的SequenceClassifierOutput不是简单的元组
        # 其中loss与logits封装方式不同，使用以下代码取出需要
        # loss = outputs.loss["loss"]
        loss = outputs[0]
        temp_loss = loss.item()
        epoch_train_loss += loss.item()

        # 反向梯度信息
        accelerator.backward(loss)

        # 参数更新
        optimizer.step()
        scheduler.step()

        total_steps += 1
        if total_steps % 50 == 0:
            loss_list.append(temp_loss)

    print("Epoch: %d, Average training loss: %.4f" % (epoch, epoch_train_loss / steps_every_epoch))


# 验证函数
from matplotlib import pyplot as plt

test_dataset = load_from_disk(dataset_path)['test']
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)


def validation():
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)
    model.eval()
    total_eval_loss = 0
    total_test_steps = len(test_dataloader)
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # 低版本transformers库在accelerate库开启混合精度训练后，模型输出的SequenceClassifierOutput不是简单的元组
        # 其中loss与logits封装方式不同，使用以下代码取出需要
        # loss = outputs.loss["loss"]
        # logits = outputs.loss["logits"]
        loss = outputs[0]
        logits = outputs[1]
        total_eval_loss += loss.item()
        pred_flat = np.argmax(logits.detach().to('cpu').numpy(), axis=1).flatten()
        labels_flat = labels.to('cpu').numpy().flatten()
        assert len(pred_flat) == len(labels_flat)
        for i in range(len(pred_flat)):
            confusion_matrix[labels_flat[i]][pred_flat[i]] = confusion_matrix[labels_flat[i]][pred_flat[i]] + 1

    TP = np.diagonal(confusion_matrix)
    accuracy = np.sum(TP) / np.sum(confusion_matrix, axis=(0, 1))
    FN = np.sum(confusion_matrix, axis=1) - TP
    FP = np.sum(confusion_matrix, axis=0) - TP
    precision = np.nan_to_num(TP / (TP + FP))
    recall = np.nan_to_num(TP / (TP + FN))
    F1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_F1_score = np.mean(F1_score)
    weighted_weight = np.array([35, 402, 392, 57, 71, 1840, 164, 114, 199, 337, 217, 124, 44]) / 3996
    weighted_precision = np.sum(precision * weighted_weight)
    weighted_recall = np.sum(recall * weighted_weight)
    weighted_F1_score = np.sum(F1_score * weighted_weight)

    aver_test_loss = total_eval_loss / total_test_steps
    print("-------------------------------")
    print("Average testing loss: %.4f" % aver_test_loss)
    print("Accuracy: %.4f" % accuracy)
    print("macro Precision: %.4f" % macro_precision)
    print("macro Recall: %.4f" % macro_recall)
    print("macro F1 Score: %.4f" % macro_F1_score)
    print("weighted Precision: %.4f" % weighted_precision)
    print("weighted Recall: %.4f" % weighted_recall)
    print("weighted F1 Score: %.4f" % weighted_F1_score)
    print("-------------------------------")
    # 转化为可以直接粘贴到word的表格的格式
    result = "%.4f\t" % aver_test_loss + "%.4f\t" % accuracy + "%.4f\t" % macro_precision + "%.4f\t" % macro_recall + "%.4f\t" % macro_F1_score + "%.4f\t" % weighted_precision + "%.4f\t" % weighted_recall + "%.4f\t" % weighted_F1_score
    print(result)
    with open("result.txt", "a") as f:
        f.write(result + "\n")
    print("-------------------------------")
    print("confusion_matrix: ")
    for i in range(num_labels):
        for j in range(num_labels):
            if i == j:
                print("%4d*" % confusion_matrix[i][j], end='\t')
            else:
                print("%5d" % confusion_matrix[i][j], end='\t')
        print()
    print("-------------------------------")

    title = model_path.replace("/", "_") + "_" + classifier_method + dataset_path[-14:] + '_train_loss'
    plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(range(len(loss_list)), loss_list)
    plt.xticks(range(len(loss_list)), [i * 50 for i in range(len(loss_list))], rotation=90)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title + '.png')
    plt.show()


if __name__ == '__main__':
    for epoch in range(epoch_num):
        print("------------Epoch: %d ----------------" % epoch)
        train()

    validation()
