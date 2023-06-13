import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class BertLastFourLayer(nn.Module):

    def __init__(self, model_path, num_labels, classifier_method="default"):
        super().__init__()
        self.num_labels = num_labels
        self.classifier_method = classifier_method
        self.config = BertConfig.from_pretrained(model_path)
        self.config.update({'output_hidden_states': True})
        self.classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.model = BertModel.from_pretrained(model_path, config=self.config)
        if self.classifier_method == "concat":
            self.classifier = nn.Linear(4 * self.config.hidden_size, num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.dropout = nn.Dropout( self.classifier_dropout)

    def forward(self, input_ids, attention_mask, labels):

        model_outputs = self.model(input_ids, attention_mask)
        all_hidden_states = torch.stack(model_outputs[2])
        all_cls_vectors = all_hidden_states[:, :, 0, :]
        last_four_cls_vectors = torch.stack(
            [all_cls_vectors[-1], all_cls_vectors[-2], all_cls_vectors[-3], all_cls_vectors[-4]], 1)
        if self.classifier_method == "concat":
            concat_cls_vector = torch.reshape(last_four_cls_vectors, (last_four_cls_vectors.shape[0], -1))
            pooled_output = concat_cls_vector
        elif self.classifier_method == "max":
            max_cls_vector = torch.max(last_four_cls_vectors, dim=1)[0]
            pooled_output = max_cls_vector
        elif self.classifier_method == "mean":
            mean_cls_vector = torch.mean(last_four_cls_vectors, dim=1)
            pooled_output = mean_cls_vector
        else:
            pooled_output = all_cls_vectors[-1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions
        )
