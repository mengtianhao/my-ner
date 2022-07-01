import logging
import os
from copy import deepcopy
import torch
import torch.nn as nn
from Model.BasicBert.BertEmbedding import BertEmbeddings
from Model.Transformer.MyTransformer import MyMultiheadAttention
from torch.nn.init import normal_


def get_activation(activation_string):
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation: %s" % act)


# 实现Bert中的多头自注意力机制
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
            MultiHeadAttention = nn.MultiheadAttention
        else:
            MultiHeadAttention = MyMultiheadAttention
        self.multi_head_attention = MultiHeadAttention(embed_dim=config.hidden_size,
                                                       num_heads=config.num_attention_heads,
                                                       dropout=config.attention_probs_dropout_prob)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


# 层Dropout、标准化和残差连接三个操作
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
            :param hidden_states: [src_len, batch_size, hidden_size]
            :param input_tensor: [src_len, batch_size, hidden_size]
            :return: [src_len, batch_size, hidden_size]
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.selfAttention = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return: [src_len, batch_size, hidden_size]
        """
        self_outputs = self.selfAttention(hidden_states, hidden_states, hidden_states, attn_mask=None,
                                          key_padding_mask=attention_mask)
        # self_outputs[0] shape: [src_len, batch_size, hidden_size]
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


# 对于`BertIntermediate`来说也就是一个普通的全连接层
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, intermediate_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, intermediate_size]
        if self.intermediate_act_fn is None:
            hidden_states = hidden_states
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        :param hidden_states: [src_len, batch_size, intermediate_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, hidden_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_attention = BertAttention(config)
        self.bert_intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len] mask掉padding部分的内容
        :return: [src_len, batch_size, hidden_size]
        """
        attention_output = self.bert_attention(hidden_states, attention_mask)
        # [src_len, batch_size, hidden_size]
        intermediate_output = self.bert_intermediate(attention_output)
        # [src_len, batch_size, intermediate_size]
        layer_output = self.bert_output(intermediate_output, attention_output)
        # [src_len, batch_size, hidden_size]
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return:
        """
        all_encoder_layers = []
        layer_output = hidden_states
        for i, layer_module in enumerate(self.bert_layers):
            layer_output = layer_module(layer_output, attention_mask)
            #  [src_len, batch_size, hidden_size]
            all_encoder_layers.append(layer_output)
        return all_encoder_layers


# 在将`BertEncoder`部分的输出结果输入到下游任务前，需要将其进行略微的处理
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        """
        :param hidden_states:  [src_len, batch_size, hidden_size]
        :return: [batch_size, hidden_size]
        """
        if self.config.pooler_type == "first_token_transform":
            token_tensor = hidden_states[0, :].reshape(-1, self.config.hidden_size)
        elif self.config.pooler_type == "all_token_average":
            token_tensor = torch.mean(hidden_states, dim=0)
        pooled_output = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        pooled_output = self.activation(pooled_output)
        # [batch_size, hidden_size]
        return pooled_output


class BertModel(nn.Module):
    # 基本的Bert模型
    def __init__(self, config):
        super().__init__()
        self.bert_embeddings = BertEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.bert_pooler = BertPooler(config)
        self.config = config
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        """
        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len] mask掉padding部分的内容
        :param token_type_ids: [src_len, batch_size]  # 如果输入模型的只有一个序列，那么这个参数也不用传值
        :param position_ids: [1,src_len] # 在实际建模时这个参数其实可以不用传值
        :return:
        """
        embedding_output = self.bert_embeddings(input_ids=input_ids, position_ids=position_ids,
                                                token_type_ids=token_type_ids)
        # embedding_output: [src_len, batch_size, hidden_size]
        all_encoder_outputs = self.bert_encoder(embedding_output, attention_mask=attention_mask)
        # all_encoder_outputs 为一个包含有num_hidden_layers个层的输出
        sequence_output = all_encoder_outputs[-1]  # 取最后一层
        # sequence_output: [src_len, batch_size, hidden_size]
        pooled_output = self.bert_pooler(sequence_output)
        # 默认是最后一层的first token 即[cls]位置经dense + tanh 后的结果
        # pooled_output: [batch_size, hidden_size]
        return pooled_output, all_encoder_outputs

    @classmethod
    def from_pretrained(cls, config, pretrained_model_dir=None):
        # 初始化模型，即一个未实例化的BertModel对象
        model = cls(config)
        # 检查Bert的预训练模型是否存在
        pretrained_model_path = os.path.join(pretrained_model_dir, 'pytorch_model.bin')
        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"<路径：{pretrained_model_path} 中的模型不存在，请仔细检查！>")
        # 预训练模型的参数
        loaded_paras = torch.load(pretrained_model_path)
        # 本地网络模型的参数
        state_dict = deepcopy(model.state_dict())
        # 预训练模型的最后八个参数不要
        loaded_paras_names = list(loaded_paras.keys())[:-8]
        # 本地模型的第一个参数不需要
        model_paras_names = list(state_dict.keys())[1:]
        # 参数初始化
        for i in range(len(loaded_paras_names)):
            logging.debug(f"## 成功将参数:{loaded_paras_names[i]}赋值给{model_paras_names[i]},"
                          f"参数形状为:{state_dict[model_paras_names[i]].size()}")
            if "position_embeddings" in model_paras_names[i]:
                # 这部分代码用来消除预训练模型只能输入小于512个字符的限制
                if config.max_position_embeddings > 512:
                    new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                         loaded_paras[loaded_paras_names[i]])
                    state_dict[model_paras_names[i]] = new_embedding
                    continue
            state_dict[model_paras_names[i]] = loaded_paras[loaded_paras_names[i]]
        logging.info(f"## 注意，正在使用本地MyTransformer中的MyMultiHeadAttention实现")
        model.load_state_dict(state_dict)
        return model


def replace_512_position(init_embedding, loaded_embedding):
    """
    本函数的作用是当max_positional_embedding > 512时，用预训练模型中的512个向量来
    替换随机初始化的positional embedding中的前512个向量
    :param init_embedding:  初始化的positional embedding矩阵，大于512行
    :param loaded_embedding: 预训练模型中的positional embedding矩阵，等于512行
    :return: 前512行被替换后的初始化的positional embedding矩阵
    """
    logging.info(f"模型参数max_positional_embedding > 512，采用替换处理！")
    init_embedding[:512, :] = loaded_embedding[:512, :]
    return init_embedding





