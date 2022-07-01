import os
import time
import torch
import logging
from Config.Model_config import Model_config
from Model.Task.BertForNER import BertForNER
from utils.data_helpers import LoadDataset


def train(config):
    model = BertForNER(config, config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()
    data_loader = LoadDataset(
        vocab_path=config.vocab_path,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=config.is_sample_shuffle
    )
    train_iter, val_iter, test_iter = data_loader.load_train_val_test_data(config.zh_msra_train_file_path,
                                                                            config.zh_msra_val_file_path,
                                                                            config.zh_msra_test_file_path)
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(config.device)
            label = label.to(config.device)
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(input_ids=sample,
                                 attention_mask=padding_mask,
                                 token_type_ids=None,
                                 position_ids=None,
                                 labels=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            label = label.transpose(0, 1).to('cpu').numpy()
            acc = 0.0
            for i in range(len(label)):
                true_label = label[i]
                true_label = torch.tensor(true_label[true_label != 0])
                predict_label = torch.tensor(logits[i])
                acc += (predict_label == true_label).float().mean()
            acc = acc / len(label)
            # acc = (logits == label).float().mean()
            if idx % 10 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            acc = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
            logging.info(f"Accuracy on val {acc:.3f}")
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), model_save_path)


def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        acc, n = 0.0, 0
        for sample, label in data_iter:
            sample = sample.to(device)
            label = label.to(device)
            padding_mask = (sample == PAD_IDX).transpose(0, 1)
            logits = model(input_ids=sample,
                           attention_mask=padding_mask,
                           labels=label,
                           is_test=True)
            label = label.transpose(0, 1).to('cpu').numpy()
            for i in range(len(label)):
                true_label = label[i]
                true_label = torch.tensor(true_label[true_label != 0])
                predict_label = torch.tensor(logits[i])
                acc += (predict_label == true_label).float().sum().item()
            n += len(label)
        model.train()
        return acc / n


if __name__ == '__main__':
    config = Model_config()
    train(config)