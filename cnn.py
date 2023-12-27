import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# 只使用每个集合的前1000个示例，以节省时间，可以根据需要更改这个数字。
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

# 初始化CNN网络的各个层
conv = Conv3x3(8)  # 数据变化：28x28x1 的图像 -> 26x26x8   这个卷积层会应用 8 个不同的 3x3 的卷积核输出 8 个特征图
pool = MaxPool2()  # 数据变化：26x26x8 的图像 -> 13x13x8  作用：减少特征图的维度，同时保留重要的信息
softmax = Softmax(13 * 13 * 8, 10)  # 数据变化：从 13x13x8 的图像到 10个输出类别    这是全连接层和softmax激活层


def forward(image, label):
    """
    完成CNN的前向传播，并计算准确率和交叉熵损失。
    - image 是一个2d numpy数组
    - label 是一个数字标签
    """
    # 将图像从 [0, 255] 转换为 [-0.5, 0.5]，以便更容易处理。这是一种标准做法。
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)
    # 使用交叉熵作为损失函数
    # 计算交叉熵损失和准确率。np.log() 是自然对数。
    loss = -np.log(out[label])
    # 看看有没有预测对
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(im, label, lr=0.005):
    """
    在给定的图像和标签上完成一个完整的训练步骤。
    返回交叉熵损失和准确率。
    - im 是一个2d numpy数组
    - label 是一个数字标签
    - lr 是学习率
    """
    # 前向传播
    out, loss, acc = forward(im, label)

    # 计算初始梯度
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # 反向传播
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc


print("MNIST CNN 初始化完成！")

# 训练CNN共3个epoch
for epoch in range(3):
    print("--- 第 %d 个epoch ---" % (epoch + 1))

    # 打乱训练数据
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # 开始训练
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 99:
            print(
                "[步骤 %d] 最近100步: 平均损失 %.3f | 准确率: %d%%"
                % (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# 测试CNN
print("\n--- 测试CNN ---")
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print("测试损失:", loss / num_tests)
print("测试准确率:", num_correct / num_tests)
