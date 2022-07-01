import torch
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
from Config.Model_config import Model_config


# 建立词典
class Vocab:
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        # 字典，可以进行索引
        self.stoi = {}
        # 列表
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            #  enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __get_itos__(self):
        return self.itos

    def __getitem__(self, token):
        # 默认值为‘[UNK]’
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


# 实例化一个词表
def build_vocab(vocab_path):
    return Vocab(vocab_path)


# 获取训练集的标签种类数
def get_all_labels(file_path):
    raw_iter = open(file_path, encoding='utf-8').readlines()
    labels = list()
    for raw in raw_iter:
        line = raw.rstrip('\n').split(" ")
        if len(line) != 1:
            labels.append(line[1])
    labels_set = set(labels)
    print(list(labels_set))


# 建立标签字典
def build_label2id():
    labels = ['[PAD]', '[CLS]', '[SEP]', 'O', 'B-NR', 'M-NR', 'E-NR', 'S-NR', 'B-NS', 'M-NS', 'E-NS', 'S-NS', 'B-NT', 'M-NT', 'E-NT', 'S-NT']
    label2id = {}
    for i in range(len(labels)):
        label2id[labels[i]] = i
    # print(label2id)
    return label2id


# 加载数据集
class LoadDataset:
    def __init__(self,
                 vocab_path='./vocab.txt',
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True):
        self.vocab = build_vocab(vocab_path)
        self.label2id = build_label2id()
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.batch_size = batch_size
        self.max_position_embeddings = max_position_embeddings
        # 判断传过来样本的最大长度
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    # 转换为token序列
    def data_process(self, file_path):
        print(file_path)
        raw_iter = open(file_path, encoding='utf-8').readlines()
        data = []
        max_len = 0
        # 获得所有的句子和对应的标签列表
        sentences, labels = list(), list()
        s, l = list(), list()
        for raw in raw_iter:
            line = raw.rstrip('\n').split(" ")
            # s, l = line[0], line[1]
            if len(line) != 1:
                s.append(line[0])
                l.append(line[1])
            else:
                sentences.append(s)
                labels.append(l)
                s, l = list(), list()
        # ncols: 调整进度条宽度, 默认是根据环境自动调节长度, 如果设置为0, 就没有进度条, 只有输出的信息
        for i in tqdm(range(len(sentences)), ncols=80):
            sentence, label = sentences[i], labels[i]
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in sentence]
            tmp_l = [self.label2id['[CLS]']] + [self.label2id[token] for token in label]
            # BERT预训练模型限制在512个字符
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]
                tmp_l = tmp_l[:self.max_position_embeddings - 1]
            tmp += [self.SEP_IDX]
            tmp_l += [self.label2id['[SEP]']]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(tmp_l, dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len

    # 对每个batch的Token序列进行padding处理
    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = pad_sequence(batch_label,
                                   padding_value=self.label2id['[PAD]'],
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        return batch_sentence, batch_label

    # 构造DataLoader迭代器
    def load_train_val_test_data(self, train_file_path=None, val_file_path=None, test_file_path=None, only_test=False):
        test_data, _ = self.data_process(test_file_path)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(train_file_path)
        # 当`max_sen_len = 'same'`时，以整个数据集中最长样本为标准，对其它进行padding；
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(val_file_path)

        train_iter = DataLoader(train_data, batch_size=self.batch_size,  shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        return train_iter, val_iter, test_iter


# padding处理
def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


if __name__ == '__main__':
    config = Model_config()
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
        print(sample.shape)  # [seq_len,batch_size]
        print(sample.transpose(0, 1))
        padding_mask = (sample == load_dataset.PAD_IDX).transpose(0, 1)
        print(padding_mask)
        print(label.shape)
        print(label)
        break


