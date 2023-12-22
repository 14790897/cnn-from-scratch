import numpy as np

class Softmax:
    # 全连接层，同时使用softmax激活函数。

    def __init__(self, input_len, nodes):
        # 我们通过输入长度来除以权重，以减少初始值的方差
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        """
        使用给定的输入执行softmax层的前向传播。
        返回一个包含相应概率值的一维numpy数组。
        - 输入可以是任何维度的数组。
        """
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        """
        执行softmax层的反向传播。
        返回这一层输入的损失梯度。
        这个求导过程比较复杂
        - d_L_d_out是这一层输出的损失梯度。
        - learn_rate是一个浮点数。
        """
        # 我们知道d_L_d_out中只有1个元素会是非零值
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # 所有e^totals的总和
            S = np.sum(t_exp)

            # out[i]关于totals的梯度
            d_out_d_t = -t_exp[i] * t_exp / (S**2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S**2)

            # totals关于权重/偏置/输入的梯度
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # 损失关于totals的梯度
            d_L_d_t = gradient * d_out_d_t

            # 损失关于权重/偏置/输入的梯度
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # 更新权重/偏置
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)
