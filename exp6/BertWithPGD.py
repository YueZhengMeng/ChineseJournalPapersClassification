import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class BertWithPGD(nn.Module):

    def __init__(self, model_path, num_labels, epsilon=1.0, alpha=0.3, k=2, emb_name='word_embeddings'):
        super().__init__()
        self.num_labels = num_labels
        self.model = BertModel.from_pretrained(model_path)
        self.config = BertConfig.from_pretrained(model_path)
        self.emb_backup = {}  # 用于保存添加扰动前的参数
        self.grad_backup = {}  # 用于保存添加扰动前的梯度
        self.epsilon = epsilon
        self.alpha = alpha
        self.K = k  # PGD attack times
        self.emb_name = emb_name
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels):
        model_outputs = self.model(input_ids, attention_mask)
        pooled_output = model_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=model_outputs.hidden_states,
                                        attentions=model_outputs.attentions)

    def attack(self,backup=False):
        # 生成扰动和对抗样本
        for name, param in self.model.named_parameters():  # 遍历模型的所有参数
            if param.requires_grad and self.emb_name in name:  # 只取word embedding层的参数
                if backup:
                    self.emb_backup[name] = param.data.clone()  # 只有第一轮添加扰动需要保存原始参数
                # 计算扰动，并在输入参数值上添加扰动
                norm_g = torch.norm(param.grad)
                if norm_g != 0 and not torch.isnan(norm_g):
                    r_at = self.alpha * param.grad / norm_g
                # 如果扰动走出了扰动半径为epsilon的空间，就映射回“球面”上，以保证扰动不要过大
                    norm_r = torch.norm(r_at)
                    if norm_r > self.epsilon:
                        r_at = self.epsilon * r_at / norm_r
                    param.data.add_(r_at)

    def restore_embedding(self):
        # 恢复添加扰动之前的参数
        for name, param in self.model.named_parameters():  # 遍历模型的所有参数
            if param.requires_grad and self.emb_name in name:  # 只取word embedding层的参数
                assert name in self.emb_backup
                param.data = self.emb_backup[name]  # 恢复添加扰动之前的参数
        self.emb_backup = {}

    def backup_grad(self):
        # 保存添加扰动之前的梯度
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        # 恢复添加扰动之前的梯度
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}
