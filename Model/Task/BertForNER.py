import torch
import torch.nn as nn
from Model.BasicBert.Bert import BertModel
from Model.Classifier.MyCRF import MyCRF
from Config.Model_config import Model_config
from utils.data_helpers import LoadDataset
import numpy as np


class BertForNER(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForNER, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.labels_num)
        self.crf = MyCRF(config.labels_num)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, labels=None, is_test=False):
        """
        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len]
        :param token_type_ids: 句子分类时为None
        :param position_ids: [1,src_len]
        :param labels: [batch_size,]
        :return:
        """
        pooled_output, all_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids)  # [batch_size,hidden_size]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.classifier(pooled_output)  # [batch_size, num_label]
        pooled_output = pooled_output.transpose(0, 1)
        labels = labels.transpose(0, 1)
        label_mask = ~attention_mask
        if not is_test:
            loss, labels = self.crf(pooled_output, labels, label_mask, is_test)
            return loss, labels
        else:
            labels = self.crf(pooled_output, labels, label_mask, is_test)
            return labels


if __name__ == '__main__':
    config = Model_config()
    model = BertForNER(config, config.pretrained_model_dir)
    model = model.to(config.device)
    load_dataset = LoadDataset(
        vocab_path=config.vocab_path,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=config.is_sample_shuffle
    )
    train_iter, val_iter, test_iter = load_dataset.load_train_val_test_data(config.zh_msra_train_file_path,
                                                                            config.zh_msra_val_file_path,
                                                                            config.zh_msra_test_file_path)
    for sample, label in train_iter:
        sample = sample.to(config.device)
        label = label.to(config.device)
        padding_mask = (sample == load_dataset.PAD_IDX).transpose(0, 1)
        loss, labels = model(input_ids=sample,
                             attention_mask=padding_mask,
                             token_type_ids=None,
                             position_ids=None,
                             labels=label)
        print(loss)
        print(labels)
        label = label.transpose(0, 1).to('cpu').numpy()
        acc = 0.0
        for i in range(len(label)):
            true_label = label[i]
            true_label = torch.tensor(true_label[true_label != 0])
            predict_label = torch.tensor(labels[i])
            acc += (predict_label == true_label).float().mean()
            # print((predict_label == true_label))
            # print(len(predict_label))
            # print((predict_label == true_label).float().sum().item())
        acc = acc / len(label)
        print(acc)
        break