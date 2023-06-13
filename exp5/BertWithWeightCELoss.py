import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class BertWithWeightCELoss(nn.Module):

    def __init__(self, model_path, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.config = BertConfig.from_pretrained(model_path)
        self.classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.model = BertModel.from_pretrained(model_path)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.class_weight = np.array([35, 402, 392, 57, 71, 1840, 164, 114, 199, 337, 217, 124, 44]) / 3996
        self.inverse_weight = (1 - self.class_weight) / (self.num_labels - 1)
        self.CE_weight = torch.from_numpy(self.inverse_weight).float()
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.loss_fn = CrossEntropyLoss(weight=self.CE_weight)

    def forward(self, input_ids, attention_mask, labels):
        model_outputs = self.model(input_ids, attention_mask)
        pooled_output = model_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions
        )
