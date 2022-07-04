import os.path
import torch
import logging
from utils.log_helper import logger_init
from Model.BasicBert.BertConfig import BertConfig
from torch.utils.tensorboard import SummaryWriter


class Model_config:
    def __init__(self, data_type):
        # 项目路径
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 预训练模型路径
        self.pretrained_model_dir = os.path.join(self.project_dir, 'bert-base-chinese')
        # Bert词典的路径
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        # 选择设备
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 数据集
        self.data_type = data_type
        if self.data_type == 'zh_msra':
            # zh_msra数据集文件夹路径
            self.data_dir = os.path.join(self.project_dir, 'Data', 'zh_msra')
            # 训练集路径
            self.train_file_path = os.path.join(self.data_dir, 'train.char.bmes')
            # 验证集路径
            self.val_file_path = os.path.join(self.data_dir, 'dev.char.bmes')
            # 测试集路径
            self.test_file_path = os.path.join(self.data_dir, 'test.char.bmes')
            # 数据集的标签列表
            self.labels_name = ['O', 'B-NR', 'M-NR', 'E-NR', 'S-NR', 'B-NS', 'M-NS', 'E-NS', 'S-NS', 'B-NT', 'M-NT',
                                'E-NT', 'S-NT']
            # 模型保存路径
            self.model_save_dir = os.path.join(self.project_dir, 'cache', 'zh_msra')
            # 标签种类数
            self.labels_num = 16
            # tensorboard文件目录名称
            self.runs_name = 'zh_msra'
            # 日志目录路径
            self.log_dir_name = 'zh_msra'
            # 日志文件名称
            self.log_file_name = 'msra'

        elif self.data_type == 'zh_ontonotes4':
            # zh_ontonotes4数据集文件夹路径
            self.data_dir = os.path.join(self.project_dir, 'Data', 'zh_ontonotes4')
            # 训练集路径
            self.train_file_path = os.path.join(self.data_dir, 'train.char.bmes')
            # 验证集路径
            self.val_file_path = os.path.join(self.data_dir, 'dev.char.bmes')
            # 测试集路径
            self.test_file_path = os.path.join(self.data_dir, 'test.char.bmes')
            # 数据集的标签列表
            self.labels_name = ['O', 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC', 'B-PER', 'M-PER', 'E-PER', 'S-PER', 'B-ORG', 'M-ORG',
                                'E-ORG', 'S-ORG', 'B-GPE', 'M-GPE', 'E-GPE', 'S-GPE']
            # 模型保存路径
            self.model_save_dir = os.path.join(self.project_dir, 'cache', 'zh_ontonotes4')
            # 标签种类数
            self.labels_num = 20
            # tensorboard文件目录名称
            self.runs_name = 'zh_ontonotes4'
            # 日志目录路径
            self.log_dir_name = 'zh_ontonotes4'
            # 日志文件名称
            self.log_file_name = 'ontonotes4'

        #
        # 是否打乱数据集，仅针对训练集
        self.is_sample_shuffle = True
        # 批处理大小
        self.batch_size = 16
        # 句子的最大长度
        self.max_sen_len = None
        # 训练次数（epoch次数）
        self.epochs = 10
        # 下接结构
        self.num_layers = 1  # 下游层数
        self.drop_prob = 0.1  # drop_out率
        # bilstm
        self.lstm_hidden = 256
        # TensorBoard神经网络可视化工具
        self.writer = SummaryWriter(os.path.join(self.project_dir, 'runs', self.runs_name))
        #
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        # 日志保存路径
        self.logs_save_dir = os.path.join(self.project_dir, 'logs', self.log_dir_name)
        # 验证模型效果的间隔
        self.model_val_per_epoch = 1
        # 日志初始化
        logger_init(log_file_name=self.log_file_name, log_level=logging.INFO, log_dir=self.logs_save_dir)
        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


if __name__ == '__main__':
    config = Model_config(data_type='zh_ontonotes4')
    print(config.train_file_path)
    print(config.labels_name)