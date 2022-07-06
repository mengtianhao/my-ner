import os
import torch
import logging

from sklearn.metrics import classification_report

from Config.Model_config import Model_config
from utils.data_helpers import LoadDataset
from Model.Task.BertForNER import BertForNER
from main import evaluate
import warnings
warnings.filterwarnings("ignore")


# 使用训练好的模型进行推理
def inference(config):
    model = BertForNER(config, config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测......")
    model = model.to(config.device)
    data_loader = LoadDataset(
        vocab_path=config.vocab_path,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=config.is_sample_shuffle,
        labels_name=config.labels_name)
    test_iter = data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                                     only_test=True)
    acc, pred_tags, true_tags = evaluate(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
    logging.info(f"Acc on test:{acc:.3f}")
    # logging loss, f1 and report
    target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    evaluation_dict = classification_report(true_tags, pred_tags, digits=4, output_dict=True)
    precision = 0
    recall = 0
    f1 = 0
    for key in evaluation_dict.keys():
        if key in target_names:
            precision += evaluation_dict[key]['precision']
            recall += evaluation_dict[key]['recall']
            f1 += evaluation_dict[key]['f1-score']
    f1 = f1 / len(target_names)
    precision = precision / len(target_names)
    recall = recall / len(target_names)
    print('precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(precision, recall, f1))


if __name__ == '__main__':
    model_config = Model_config(data_type='zh_ontonotes4')
    inference(model_config)