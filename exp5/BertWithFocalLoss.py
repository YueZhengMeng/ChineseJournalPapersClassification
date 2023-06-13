import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class MultiFocalLoss(nn.Module):

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()
            prob = prob.view(-1, prob.size(-1))

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth
        logpt = torch.log(prob)
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


class BertWithFocalLoss(nn.Module):

    def __init__(self, model_path, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.config = BertConfig.from_pretrained(model_path)
        self.classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.model = BertModel.from_pretrained(model_path)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.loss_fn = MultiFocalLoss(self.num_labels)

    def forward(self, input_ids, attention_mask, labels):
        model_outputs = self.model(input_ids, attention_mask)
        pooled_output = model_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions)
