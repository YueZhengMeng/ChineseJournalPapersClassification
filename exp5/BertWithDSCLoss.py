import torch
from torch import nn
from transformers import BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class MultiDSCLoss(torch.nn.Module):

    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.smooth) / (probs_with_factor + 1 + self.smooth)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")


class BertWithDSCLoss(nn.Module):

    def __init__(self, model_path, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.config = BertConfig.from_pretrained(model_path)
        self.classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.model = BertModel.from_pretrained(model_path)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.loss_fn = MultiDSCLoss()

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
