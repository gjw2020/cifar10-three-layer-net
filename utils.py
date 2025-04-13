import numpy as np
from matplotlib import pyplot as plt
import pickle


# relu激活函数
def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)


# relu导数
def relu_derivative(x):
    """ReLU激活函数的导数"""
    return (x > 0).astype(float)


# sigmoid激活函数
def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))


# sigmoid导数
def sigmoid_derivative(x):
    """Sigmoid激活函数的导数"""
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


# Softmax函数
def softmax(scores):
    """Softmax函数，将分数转换为概率分布"""
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# 交叉熵损失函数
def cross_entropy_loss(probs, y):
    """计算交叉熵损失"""
    N = probs.shape[0]
    correct_logprobs = -np.log(probs[np.arange(N), y])
    return np.mean(correct_logprobs)


# 计算准确率
def compute_accuracy(model, X, y):
    """计算模型在给定数据上的准确率"""
    scores, _ = model.forward(X)
    y_pred = np.argmax(scores, axis=1)
    return np.mean(y_pred == y)


# 随机打乱数据
def shuffle_data(X, y):
    """打乱训练数据"""
    num_train = X.shape[0]
    indices = np.arange(num_train)
    np.random.shuffle(indices)
    return X[indices], y[indices]


# 训练过程可视化
def visualize_training(all_train_losses, all_val_losses, all_val_accuracies, all_param_combinations):
    epochs = range(1, len(all_train_losses[0]) + 1)

    plt.figure(figsize=(18, 6))

    # 绘制训练损失曲线
    plt.subplot(1, 3, 1)
    for i, train_losses in enumerate(all_train_losses):
        plt.plot(epochs, train_losses, label=all_param_combinations[i])
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制验证损失曲线
    plt.subplot(1, 3, 2)
    for i, val_losses in enumerate(all_val_losses):
        plt.plot(epochs, val_losses, label=all_param_combinations[i])
    plt.title('Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制验证准确率曲线
    plt.subplot(1, 3, 3)
    for i, val_accuracies in enumerate(all_val_accuracies):
        plt.plot(epochs, val_accuracies, label=all_param_combinations[i])
    plt.title('Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('combined_visualization.png')
    plt.show()


def visualize_model_parameters(model, save_path='model_parameters_visualization.png'):
    """
    可视化三层神经网络的网络参数（权重）并保存图片
    :param model: 训练好的 ThreeLayerNet 模型实例
    :param save_path: 图片保存的路径，默认为'model_parameters_visualization.png'
    """
    # 获取第一层和第二层的权重
    W1 = model.params['W1']
    W2 = model.params['W2']

    # 可视化第一层权重
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(W1, cmap='viridis', interpolation='nearest')
    plt.title('Weights of First Layer (W1)')
    plt.colorbar()
    plt.xlabel('Hidden Units')
    plt.ylabel('Input Features')

    # 可视化第二层权重
    plt.subplot(1, 2, 2)
    plt.imshow(W2, cmap='viridis', interpolation='nearest')
    plt.title('Weights of Second Layer (W2)')
    plt.colorbar()
    plt.xlabel('Output Classes')
    plt.ylabel('Hidden Units')

    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path)
    plt.show()


# 保存模型权重
def save_model(model, filename):
    """保存模型权重到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(model.params, f)


# 加载模型
def load_model(model, filename):
    """从文件中加载模型权重"""
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    model.params = params
    return model


