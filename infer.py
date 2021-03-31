import os

import numpy as np
import paddle

from utils.model import Model
from utils.data import process
from utils.decoder import ctc_greedy_decoder


with open('dataset/vocabulary.txt', 'r', encoding='utf-8') as f:
    vocabulary = f.readlines()

vocabulary = [v.replace('\n', '') for v in vocabulary]

save_model = 'models/'
model = Model(vocabulary, image_height=32)
model.set_state_dict(paddle.load(os.path.join(save_model, 'model.pdparams')))
model.eval()


def infer(path):
    data = process(path, img_height=32)
    data = data[np.newaxis, :]
    data = paddle.to_tensor(data, dtype='float32')
    # 执行识别
    out = model(data)
    out = paddle.transpose(out, perm=[1, 0, 2])
    out = paddle.nn.functional.softmax(out)[0]
    # 解码获取识别结果
    out_string = ctc_greedy_decoder(out, vocabulary)

    print('预测结果：%s' % out_string)


if __name__ == '__main__':
    image_path = 'dataset/train_data/0aw2.png'
    infer(image_path)
