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
    max_acc = 0
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        max_acc = checkpoint['max_acc']  # 加载上次的最好的正确率
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    # 不冻结的部分参数
    unfreeze_layers = ['bert_layers.11', 'bert_pooler', 'bilstm', 'crf']
    # 打印模型参数
    # for name, param in model.named_parameters():
    #     print(name, param.size())
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    # 过滤掉requires_grad = False的参数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    model.train()
    data_loader = LoadDataset(
        vocab_path=config.vocab_path,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=config.is_sample_shuffle,
        labels_name=config.labels_name
    )
    train_iter, val_iter, test_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)

    for epoch in range(config.epochs):
        losses = 0  # 总的损失率
        acces = 0   # 总的精确率
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(config.device)
            label = label.to(config.device)
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            loss, labels = model(input_ids=sample,
                                 attention_mask=padding_mask,
                                 token_type_ids=None,
                                 position_ids=None,
                                 labels=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc, _true, _n = calculate(labels, label)
            acces += acc
            if idx % 10 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        train_acc = acces / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        # 增加tensorboard显示
        config.writer.add_scalar('Training/Loss', train_loss, epoch)
        config.writer.add_scalar('Training/Accuracy', train_acc, epoch)
        #
        if (epoch + 1) % config.model_val_per_epoch == 0:
            acc, _, _1 = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
            logging.info(f"Accuracy on val {acc:.3f}")
            config.writer.add_scalar('evaluating/Accuracy', acc, epoch)
            if acc > max_acc:
                max_acc = acc
                state_dict = model.state_dict()
                torch.save({'max_acc': max_acc,
                            'model_state_dict': state_dict}, model_save_path)


def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        true, n = 0, 0
        total_true_label = []
        total_predict_label = []
        for sample, label in data_iter:
            sample = sample.to(device)
            label = label.to(device)
            padding_mask = (sample == PAD_IDX).transpose(0, 1)
            labels = model(input_ids=sample,
                           attention_mask=padding_mask,
                           labels=label,
                           is_test=True)
            _, _true, _n = calculate(labels, label)
            true += _true
            n += _n
            #
            predict_label = [j for i in labels for j in i]
            true_label = label.transpose(0, 1)
            true_label = torch.as_tensor(true_label[true_label != 0]).to('cpu').numpy()
            true_label = [i for i in true_label]
            total_predict_label = total_predict_label + predict_label
            total_true_label = total_true_label + true_label
            #
        model.train()
        return true / n, total_predict_label, total_true_label


# 计算预测精确度，按实体计算
def calculate(predict_labels, true_labels):
    true_labels = true_labels.transpose(0, 1).to('cpu').numpy()
    acc, true, n = 0.0, 0, 0
    for i in range(len(true_labels)):
        true_label = true_labels[i]
        # 去掉真实标签中的padding项
        true_label = torch.tensor(true_label[true_label != 0])
        predict_label = torch.tensor(predict_labels[i])
        # 去掉预测标签和真实标签中的'[CLS]', '[SEP]', 'O'项,分别在字典中显示为1，2，3
        new_true_label = []
        new_predict_label = []
        for j in range(len(true_label)):
            if true_label[j] != torch.tensor(1):
                if true_label[j] != torch.tensor(2):
                    if true_label[j] != torch.tensor(3):
                        new_true_label.append(true_label[j])
                        new_predict_label.append(predict_label[j])
        true_label = torch.tensor(new_true_label)
        predict_label = torch.tensor(new_predict_label)
        true += (predict_label == true_label).float().sum().item()
        n += len(predict_label)
    if n == 0:
        acc = 1.0
    else:
        acc = true / n
    return acc, true, n


if __name__ == '__main__':
    config = Model_config(data_type='zh_ontonotes4')
    train(config)