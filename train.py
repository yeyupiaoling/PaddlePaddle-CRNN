import paddle
import numpy as np
import os
from datetime import datetime
from utils.model import Model
from utils.decoder import ctc_greedy_decoder, label_to_string, cer
from paddle.io import DataLoader
from utils.data import collate_fn
from utils.data import CustomDataset
from visualdl import LogWriter

# 训练数据列表路径
train_data_list_path = 'dataset/train_list.txt'
# 测试数据列表路径
test_data_list_path = 'dataset/test_list.txt'
# 词汇表路径
voc_path = 'dataset/vocabulary.txt'
# 模型保存的路径
save_model = 'models/'
# 每一批数据大小
batch_size = 32
# 预训练模型路径
pretrained_model = None
# 训练轮数
num_epoch = 100
# 初始学习率大小
learning_rate = 1e-3
# 日志记录噐
writer = LogWriter(logdir='log')


def train():
    # 获取训练数据
    train_dataset = CustomDataset(train_data_list_path, voc_path, img_height=32)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    # 获取测试数据
    test_dataset = CustomDataset(test_data_list_path, voc_path, img_height=32, is_data_enhance=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    # 获取模型
    model = Model(train_dataset.vocabulary, image_height=train_dataset.img_height, channel=1)
    paddle.summary(model, input_size=(batch_size, 1, train_dataset.img_height, 500))
    # 设置优化方法
    boundaries = [30, 100, 200]
    lr = [0.1 ** l * learning_rate for l in range(len(boundaries) + 1)]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=lr, verbose=False)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=scheduler,
                                      weight_decay=paddle.regularizer.L2Decay(1e-4))
    # 获取损失函数
    ctc_loss = paddle.nn.CTCLoss()
    # 加载预训练模型
    if pretrained_model is not None:
        model.set_state_dict(paddle.load(os.path.join(pretrained_model, 'model.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(pretrained_model, 'optimizer.pdopt')))

    train_step = 0
    test_step = 0
    # 开始训练
    for epoch in range(num_epoch):
        for batch_id, (inputs, labels, input_lengths, label_lengths) in enumerate(train_loader()):
            out = model(inputs)
            # 计算损失
            input_lengths = paddle.full(shape=[batch_size], fill_value=out.shape[0], dtype='int64')
            loss = ctc_loss(out, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0:
                print('[%s] Train epoch %d, batch %d, loss: %f' % (datetime.now(), epoch, batch_id, loss))
                writer.add_scalar('Train loss', loss, train_step)
                train_step += 1
        # 执行评估
        if epoch % 10 == 0:
            model.eval()
            cer = evaluate(model, test_loader, train_dataset.vocabulary)
            print('[%s] Test epoch %d, cer: %f' % (datetime.now(), epoch, cer))
            writer.add_scalar('Test cer', cer, test_step)
            test_step += 1
            model.train()
        # 记录学习率
        writer.add_scalar('Learning rate', scheduler.last_lr, epoch)
        scheduler.step()
        # 保存模型
        paddle.save(model.state_dict(), os.path.join(save_model, 'model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(save_model, 'optimizer.pdopt'))


# 评估模型
def evaluate(model, test_loader, vocabulary):
    cer_result = []
    for batch_id, (inputs, labels, _, _) in enumerate(test_loader()):
        # 执行识别
        outs = model(inputs)
        outs = paddle.transpose(outs, perm=[1, 0, 2])
        outs = paddle.nn.functional.softmax(outs)
        # 解码获取识别结果
        labelss = []
        out_strings = []
        for out in outs:
            out_string = ctc_greedy_decoder(out, vocabulary)
            out_strings.append(out_string)
        for i, label in enumerate(labels):
            label_str = label_to_string(label, vocabulary)
            labelss.append(label_str)
        for out_string, label in zip(*(out_strings, labelss)):
            # 计算字错率
            c = cer(out_string, label) / float(len(label))
            cer_result.append(c)
    cer_result = float(np.mean(cer_result))
    return cer_result


if __name__ == '__main__':
    train()
