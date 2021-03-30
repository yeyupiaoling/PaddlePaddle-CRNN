
import Levenshtein as Lev
from itertools import groupby
import paddle


def ctc_greedy_decoder(probs_seq, vocabulary, blank=0):
    """CTC贪婪（最佳路径）解码器。
    由最可能的令牌组成的路径被进一步后处理
    删除连续的重复和所有的空白。
    :param probs_seq: 每个词汇表上概率的二维列表字符。
                      每个元素都是浮点概率列表为一个字符。
    :type probs_seq: list
    :param vocabulary: 词汇表
    :type vocabulary: list
    :param blank: 空白索引
    :type blank: int
    :return: 解码结果字符串
    :rtype: baseline
    """
    # 尺寸验证
    for probs in probs_seq:
        if not len(probs) == len(vocabulary):
            raise ValueError("probs_seq 尺寸与词汇不匹配")
    # argmax以获得每个时间步长的最佳指标
    max_index_list = paddle.argmax(probs_seq, -1).numpy()
    # 删除连续的重复索引
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # 删除空白索引
    index_list = [index for index in index_list if index != blank]
    # 将索引列表转换为字符串
    return ''.join([vocabulary[index] for index in index_list])


def label_to_string(label, vocabulary, blank=0):
    """标签转文字

    :param label: 结果的标签，或者数据集的标签
    :type label: list
    :param vocabulary: 词汇表
    :type vocabulary: list
    :param blank: 空白索引
    :type blank: int
    :return: 解码结果字符串
    :rtype: baseline
    """
    label = [index for index in label if index != blank]
    return ''.join([vocabulary[index] for index in label])


def cer(out_string, target_string):
    """通过计算两个字符串的距离，得出字错率

    Arguments:
        out_string (string): 比较的字符串
        target_string (string): 比较的字符串
    """
    s1, s2, = out_string.replace(" ", ""), target_string.replace(" ", "")
    return Lev.distance(s1, s2)
