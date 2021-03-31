import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance
from paddle.io import Dataset


# 随机调整亮度，进行数据增强
def random_brightness(img):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        brightness_delta = 0.3
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


# 随机调整对比度，进行数据增强
def random_contrast(img):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        contrast_delta = 0.3
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


# 随机调整饱和度，进行数据增强
def random_saturation(img):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        saturation_delta = 0.3
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


# 随机调整色相，进行数据增强
def random_hue(img):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        hue_delta = 48
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


# 随机裁剪
def random_crop(img):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        w, h = img.size
        xmin = random.randint(0, 5)
        ymin = random.randint(0, 5)
        xmax = random.randint(w - 5, w)
        ymax = random.randint(h - 5, h)
        img = img.crop((xmin, ymin, xmax, ymax))
    return img


# 数据增强
def data_enhance(img):
    prob = np.random.uniform(0, 1)
    img = random_crop(img)
    if prob > 0.5:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    else:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


# 图像预处理
def process(path, img_height, is_data_enhance=True):
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if is_data_enhance:
        image = data_enhance(image)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # 转灰度图
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    r = h / img_height
    width = int(w / r)
    # 缩放统一高度
    image = cv2.resize(image, (width, img_height))
    # 转换成CHW
    image = image[np.newaxis, :]
    # 归一化
    image = image / 255.
    return image


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_list_path, voc_path, img_height=32, is_data_enhance=True):
        """
        数据加载器
        :param data_list_path: 数据列表路径
        :param voc_path: 词汇表路径
        :param img_height: 固定图片的高度
        """
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        with open(voc_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        self.vocabulary = [labels[i].replace('\n', '') for i in range(len(labels))]
        self.vocabulary_dict = dict([(labels[i].replace('\n', ''), i) for i in range(len(labels))])
        self.img_height = img_height
        self.is_data_enhance = is_data_enhance

    def __getitem__(self, idx):
        path, label = self.lines[idx].replace('\n', '').split('\t')
        img = process(path, self.img_height, self.is_data_enhance)
        # 将字符标签转换为int数据
        transcript = [self.vocabulary_dict.get(x) for x in label]
        img = np.array(img, dtype='float32')
        # 标签变长
        transcript = np.array(transcript, dtype='int32')
        return img, transcript

    def __len__(self):
        return len(self.lines)


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[2], reverse=True)
    channel_size = batch[0][0].shape[0]
    height_size = batch[0][0].shape[1]
    max_width_length = batch[0][0].shape[2]
    batch_size = len(batch)
    # 找出标签最长的
    batch_temp = sorted(batch, key=lambda sample: len(sample[1]), reverse=True)
    max_label_length = len(batch_temp[0][1])
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, channel_size, height_size, max_width_length), dtype='float32')
    labels = np.zeros((batch_size, max_label_length), dtype='int32')
    input_lens = []
    label_lens = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        width_length = tensor.shape[2]
        label_length = target.shape[0]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :, :, :width_length] = tensor[:, :, :]
        labels[x, :label_length] = target[:]
        input_lens.append(width_length)
        label_lens.append(len(target))
    input_lens = np.array(input_lens, dtype='int64')
    label_lens = np.array(label_lens, dtype='int64')
    return inputs, labels, input_lens, label_lens
