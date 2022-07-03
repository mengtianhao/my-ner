import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, tag_size, embedding_size, hidden_size, num_layers, dropout, with_ln):
        """
        :param tag_size: 目标向量的维度
        :param embedding_size: 输入向量的维度
        :param hidden_size: bilstm的输出向量的维度
        :param num_layers: LSTM隐藏层的层数
        :param dropout: 丢弃率
        :param with_ln: 使用层归一化
        """
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # set multi-lstm dropout
        self.multi_dropout = 0. if num_layers == 1 else dropout
        self.bilstm = nn.LSTM(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=self.multi_dropout,
                              bidirectional=True)

        # 使用层归一化
        self.with_ln = with_ln
        if with_ln:
            self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.hidden2tag.weight)

    def get_lstm_features(self, embed, mask):
        """
        :param embed: (seq_len, batch_size, embedding_size)
        :param mask: (seq_len, batch_size)
        :return lstm_features: (seq_len, batch_size, tag_size)
        """
        embed = self.dropout(embed)
        max_len, batch_size, embed_size = embed.size()
        """
        将一个经过padding后的变长序列压紧，压缩后就不含padding的字符0了
        第一步：padding后的输入序列先经过nn.utils.rnn.pack_padded_sequence，这样会得到一个PackedSequence类型的object，
              可以直接传给RNN（RNN的源码中的forward函数里上来就是判断输入是否是PackedSequence的实例，进而采取不同的操作，如果是则输出也是该类型。）；
        第二步：得到的PackedSequence类型的object，正常直接传给RNN，得到的同样是该类型的输出；
        第三步：再经过nn.utils.rnn.pad_packed_sequence，也就是对经过RNN后的输出重新进行padding操作，得到正常的每个batch等长的序列。
        """
        embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(0).long().cpu(), enforce_sorted=False)
        lstm_output, _ = self.bilstm(embed)  # (seq_len, batch_size, hidden_size*2)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, total_length=max_len)
        # (seq_len, batch_size, hidden_size*2) * (seq_len, batch_size， 1)
        # *表示两个矩阵对应位置处的两个元素相乘
        lstm_output = lstm_output * mask.unsqueeze(-1)
        if self.with_ln:
            lstm_output = self.layer_norm(lstm_output)
        lstm_features = self.hidden2tag(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, tag_size)
        return lstm_features
