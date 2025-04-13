from utils import *

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', reg=0.0):
        # 检查激活函数是否支持
        supported_activations = ['relu', 'sigmoid']
        if activation not in supported_activations:
            raise ValueError(f"Unsupported activation: {activation}. Supported activations are {supported_activations}.")

        # 根据激活函数选择合适的初始化方法
        if activation == 'relu':
            # He初始化
            scale_W1 = np.sqrt(2 / input_size)
            scale_W2 = np.sqrt(2 / hidden_size)
        elif activation == 'sigmoid':
            # Xavier初始化
            scale_W1 = np.sqrt(1 / input_size)
            scale_W2 = np.sqrt(1 / hidden_size)

        # 初始化权重和偏置
        self.params = {}
        self.params['W1'] = scale_W1 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = scale_W2 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 激活函数类型
        self.activation = activation
        # L2正则化强度
        self.reg = reg

    def forward(self, X):
        """执行一次前向传播，返回输出层的分类得分。"""
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        # 隐藏层线性变换
        z1 = X.dot(W1) + b1
        # 激活函数
        if self.activation == 'relu':
            a1 = relu(z1)
        elif self.activation == 'sigmoid':
            a1 = sigmoid(z1)
        # 输出层线性变换（未应用 softmax 前的分数）
        scores = a1.dot(W2) + b2
        return scores, a1

    def compute_loss_and_grads(self, X, y):
        """
        前向传播计算损失，然后反向传播计算梯度。
        X: 输入数据，[N x D]; y: 标签，[N]的整数类别
        返回: (loss, grads) 二元组，loss是标量，grads是字典包含 dW1, db1, dW2, db2
        """
        # 检查输入数据
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Input X must be a 2D array and y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")

        # 前向传播
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        # 得到输出分数和隐藏层激活值
        scores, a1 = self.forward(X)
        # 输出层使用 Softmax，将 scores 转换为概率
        probs = softmax(scores)
        # 计算交叉熵损失
        data_loss = cross_entropy_loss(probs, y)
        # L2正则化损失
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss

        # 反向传播计算梯度
        grads = {}
        # 输出层梯度
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N
        # W2 和 b2 的梯度
        grads['W2'] = a1.T.dot(dscores) + self.reg * W2
        grads['b2'] = np.sum(dscores, axis=0)
        # 隐藏层梯度
        dhidden = dscores.dot(W2.T)
        # 隐藏层激活函数的梯度
        if self.activation == 'relu':
            dhidden *= relu_derivative(a1)
        elif self.activation == 'sigmoid':
            dhidden *= sigmoid_derivative(a1)
        # 现在 dhidden 是对 z1 的梯度
        grads['W1'] = X.T.dot(dhidden) + self.reg * W1
        grads['b1'] = np.sum(dhidden, axis=0)
        return loss, grads