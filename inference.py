import os
import torch
import logging
from Config.Model_config import Model_config
from utils.data_helpers import LoadDataset
from Model.Task.BertForNER import BertForNER
from main import evaluate


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
    acc = evaluate(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
    logging.info(f"Acc on test:{acc:.3f}")


if __name__ == '__main__':
    model_config = Model_config(data_type='zh_ontonotes4')
    inference(model_config)