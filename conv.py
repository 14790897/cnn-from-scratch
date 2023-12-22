import numpy as np

"""
注意：在这个实现中，我们假设输入是一个2维的numpy数组，为了简化实现，因为这就是我们的MNIST图像的存储方式。这对我们来说是可行的，因为我们将它用作我们网络的第一层，但是大多数CNN有更多的卷积层。如果我们要构建一个需要多次使用Conv3x3的更大的网络，我们就必须让输入成为一个3维的numpy数组。
"""


class Conv3x3:
    # 使用3x3过滤器的卷积层。

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters是一个3维数组，其维度为(num_filters, 3, 3)表示卷积层包含了 num_filters 个大小为 3x3 的过滤器
        # 我们通过除以9来减少初始值的方差
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        """
        3*3卷积核在图像上滑动，每次滑动1个像素，使用迭代器返回每次滑动的区域
        - image是一个2维的numpy数组。
        """
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i : (i + 3), j : (j + 3)]
                yield im_region, i, j

    def forward(self, input):
        """
        使用给定的输入执行卷积层的前向传播。
        返回一个3维的numpy数组，其维度为(h, w, num_filters)。
        - input是一个2维的numpy数组
        """
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backprop(self, d_L_d_out, learn_rate):
        """
        执行卷积层的反向传播。
        - d_L_d_out是这一层输出的损失梯度。
        - learn_rate是一个浮点数。
        """
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # 更新过滤器
        self.filters -= learn_rate * d_L_d_filters

        # 我们在这里不返回任何东西，因为我们将Conv3x3用作CNN中的第一层。
        return None
