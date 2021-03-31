import os

import cv2
from tqdm import tqdm


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    # 转完之后不是半角字符返回原来的字符
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)


def str_Q2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


def create_list():
    with open('dataset/train_label.csv', 'r', encoding='gbk') as f:
        lines = f.readlines()

    # 创建训练数据列表
    max_label = 0
    max_image = 0
    list_path = 'dataset/train_list.txt'
    with open(list_path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(1, len(lines))):
            line = lines[i]
            path, label = str(line).replace('\n', '').split(',')
            # 删除空格
            label = label.replace('　', '').replace(' ', '')
            # 全角 -> 半角
            label = str_Q2B(label)
            if max_label < len(label):
                max_label = len(label)
            image_path = os.path.join('dataset/train_images', path).replace('\\', '/')
            image = cv2.imread(image_path)
            h, w, c = image.shape
            if max_image < w:
                max_image = w
            # 写入图像路径和label，用Tab隔开
            f.write(image_path + '\t' + label + '\n')
    print("最长的标签为：%d" % max_label)
    print("最长的图片为：%d" % max_image)

    # 创建测试数据列表
    list_path = 'dataset/test_list.txt'
    with open(list_path, 'w', encoding='utf-8') as f:
        for i in range(1, len(lines)):
            if i % 10 != 0:
                continue
            line = lines[i]
            path, label = str(line).replace('\n', '').split(',')
            # 删除空格
            label = label.replace('　', '').replace(' ', '')
            # 全角 -> 半角
            label = str_Q2B(label)
            image_path = os.path.join('dataset/train_images', path).replace('\\', '/')
            # 写入图像路径和label，用Tab隔开
            f.write(image_path + '\t' + label + '\n')


def creat_vocabulary():
    # 生成词汇表
    with open('dataset/train_list.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    v = set()
    for line in lines:
        _, label = line.replace('\n', '').split('\t')
        for c in label:
            v.add(c)

    vocabulary_path = 'dataset/vocabulary.txt'
    with open(vocabulary_path, 'w', encoding='utf-8') as f:
        f.write(' \n')
        for c in v:
            f.write(c + '\n')


if __name__ == '__main__':
    create_list()
    creat_vocabulary()
