import os.path
import torch
import logging
from utils.log_helper import logger_init
from Model.BasicBert.BertConfig import BertConfig
from torch.utils.tensorboard import SummaryWriter


class Model_config:
    def __init__(self):
        # 项目路径
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 预训练模型路径
        self.pretrained_model_dir = os.path.join(self.project_dir, 'bert-base-chinese')
        # Bert词典的路径
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        # 数据集路径
        self.zh_msra_dir = os.path.join(self.project_dir, 'Data', 'zh_msra')
        # 选择设备
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 训练集路径
        self.zh_msra_train_file_path = os.path.join(self.zh_msra_dir, 'train.char.bmes')
        # 验证集路径
        self.zh_msra_val_file_path = os.path.join(self.zh_msra_dir, 'dev.char.bmes')
        # 测试集路径
        self.zh_msra_test_file_path = os.path.join(self.zh_msra_dir, 'test.char.bmes')
        # 是否打乱数据集，仅针对训练集
        self.is_sample_shuffle = True
        # 标签种类数
        self.labels_num = 16
        # 批处理大小
        self.batch_size = 4
        # 句子的最大长度
        self.max_sen_len = None
        # 训练次数（epoch次数）
        self.epochs = 10
        # 项目名称
        self.data_name = 'my-ner'
        # TensorBoard神经网络可视化工具
        self.writer = SummaryWriter(f"runs/{self.data_name}")
        # 模型保存路径
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        # 日志保存路径
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        # 验证模型效果的间隔
        self.model_val_per_epoch = 1
        # 日志初始化
        logger_init(log_file_name='transformer', log_level=logging.INFO, log_dir=self.logs_save_dir)
        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")