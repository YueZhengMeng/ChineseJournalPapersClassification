import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class BertWithFGM(nn.Module):

    def __init__(self, model_path, num_labels, epsilon=1.0, emb_name='word_embeddings'):
        super().__init__()
        self.num_labels = num_labels
        self.model = BertModel.from_pretrained(model_path)
        self.config = BertConfig.from_pretrained(model_path)
        self.backup = {}  # 用于保存添加扰动前的参数
        self.epsilon = epsilon
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

    def attack(self):
        # 生成扰动和对抗样本
        for name, param in self.model.named_parameters():  # 遍历模型的所有参数
            if param.requires_grad and self.emb_name in name:  # 只取word embedding层的参数
                self.backup[name] = param.data.clone()  # 保存添加扰动之前的参数
                norm = torch.norm(param.grad)  # 对参数梯度进行第二范式归一化
                if norm != 0 and not torch.isnan(norm):  # 计算扰动，并在输入参数值上添加扰动
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # 恢复添加扰动之前的参数
        for name, param in self.model.named_parameters():  # 遍历模型的所有参数
            if param.requires_grad and self.emb_name in name:  # 只取word embedding层的参数
                assert name in self.backup
                param.data = self.backup[name]  # 恢复添加扰动之前的参数
        self.backup = {}
