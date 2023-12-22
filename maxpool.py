import numpy as np


class MaxPool2:
    # 使用2x2池大小的最大池化层。

    def iterate_regions(self, image):
        """
        生成不重叠的2x2图像区域进行池化。
        - image 是一个3维numpy数组，其形状为 (高度, 宽度, 过滤器数量)。
        """
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2) : (i * 2 + 2), (j * 2) : (j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        """
        使用给定的输入执行最大池化层的前向传播。
        返回一个3维numpy数组，其维度为 (高度 / 2, 宽度 / 2, 过滤器数量)。
        因为是2乘2的池化层，所以所返回的大小要除以二，但是不会影响过滤器数量。
        - input 是一个3维numpy数组，其维度为 (高度, 宽度, 过滤器数量)。
        """
        self.last_input = input

        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop(self, d_L_d_out):
        """
        在最大池化层的反向传播中，仅将损失梯度传递给每个池化区域（这里是2x2窗口）中的最大值所在的元素。这些元素在前向传播中对输出层有直接贡献。对于非最大值的元素，由于它们在前向传播中没有对输出产生影响，因此其梯度为零。
        简单来说就是只对池化区域中最大的那个值进行接下来的反向传播过程，这是通过将其他的值变为零完成的。
        """
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[
                                i, j, f2
                            ]

        return d_L_d_input
