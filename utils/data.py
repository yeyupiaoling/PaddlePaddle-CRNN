import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance
from paddle.io import Dataset


# 随机调整亮度，进行数据增强
def random_brightness(img):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        brightness_delta = 0.5
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


# 随机调整对比度，进行数据增强
def random_contrast(img):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        contrast_delta = 0.5
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


# 随机调整饱和度，进行数据增强
def random_saturation(img):
    prob = np.random.uniform(0, 1)
    if prob < 0.5:
        saturation_delta = 0.5
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
def process(path, img_width, img_height, is_data_enhance=True):
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
    if w < img_width:
        # 统一缩放大小
        image = cv2.resize(image, (width, img_height))
        image_temp = np.zeros((img_height, img_width - width))
        image = np.hstack((image, image_temp))
        image_length = w
    else:
        image = cv2.resize(image, (img_width, img_height))
        image_length = img_width
    # 转换成CHW
    image = image[np.newaxis, :]
    # 归一化
    image = (image - 128) / 128
    return image, image_length


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_list_path, voc_path, img_width=1000, img_height=32, max_label_length=100,
                 is_data_enhance=True):
        """
        数据加载器
        :param data_list_path: 数据列表路径
        :param voc_path: 词汇表路径
        :param img_width: 固定图片的宽度
        :param img_height: 固定图片的高度
        :param max_label_length: 固定标签的长度
        """
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        with open(voc_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
        self.vocabulary = [labels[i].replace('\n', '') for i in range(len(labels))]
        self.vocabulary_dict = dict([(labels[i].replace('\n', ''), i) for i in range(len(labels))])
        self.img_width = img_width
        self.img_height = img_height
        self.max_label_length = max_label_length
        self.is_data_enhance = is_data_enhance

    def __getitem__(self, idx):
        path, label = self.lines[idx].replace('\n', '').split('\t')
        img, img_length = process(path, self.img_width, self.img_height, self.is_data_enhance)
        img_length = np.array(img_length, dtype='int64')
        # 将字符标签转换为int数据
        transcript = [self.vocabulary_dict.get(x) for x in label]
        img = np.array(img, dtype='float32')
        # 标签变长
        transcript = np.array(transcript, dtype='int32')
        label_length = np.array(len(transcript), dtype='int64')
        # 固定标签长度
        if len(transcript) < self.max_label_length:
            zeros = np.zeros(self.max_label_length - len(transcript), dtype='int32')
            transcript = np.hstack((transcript, zeros))
        else:
            transcript = transcript[:self.max_label_length]
        return img, transcript, img_length, label_length

    def __len__(self):
        return len(self.lines)
