import paddle
import paddle.nn as nn


class Model(nn.Layer):
    def __init__(self, vocabulary, image_height, channel=1):
        super(Model, self).__init__()
        assert image_height % 32 == 0, 'image Height has to be a multiple of 32'

        self.conv1 = nn.Conv2D(in_channels=channel, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2D(256)

        self.conv4 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2D(kernel_size=(2,2), stride=(2, 1), padding=(0, 1))

        self.conv5 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2D(512)

        self.conv6 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2D(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))

        self.conv7 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0)
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm2D(512)

        self.lstm1 = nn.LSTM(input_size=512, hidden_size=256, direction='bidirectional')
        self.fc = nn.Linear(in_features=512, out_features=256)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, direction='bidirectional')

        self.output = nn.Linear(in_features=512, out_features=len(vocabulary))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.pool6(self.relu6(self.conv6(x)))
        conv = self.relu7(self.bn7(self.conv7(x)))

        # PaddlePaddle框架输出结构为BCHW
        batch, channel, height, width = conv.shape
        assert height == 1, "The output height must be 1."
        # 将height==1的维度去掉-->BCW
        conv = paddle.squeeze(conv, axis=2)

        x = paddle.transpose(conv, perm=[2, 0, 1])
        y, (h, c) = self.lstm1(x)
        t, b, h = y.shape
        x = paddle.reshape(y, shape=(t * b, h))
        x = self.fc(x)
        x = paddle.reshape(x, shape=(t, b, -1))
        y, (h, c) = self.lstm2(x)
        t, b, h = y.shape
        x = paddle.reshape(y, shape=(t * b, h))
        x = self.output(x)
        x = paddle.reshape(x, shape=(t, b, -1))
        return x
