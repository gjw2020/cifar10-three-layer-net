from data_load import *
from hyperparameter_tuning import *
from model import ThreeLayerNet
from predict import *
from train import *
from utils import *
import pickle

def predict(model, X):
    """对输入数据进行预测，返回预测的类别标签"""
    scores, _ = model.forward(X)
    y_pred = np.argmax(scores, axis=1)
    return y_pred


def test_model(model, X_test, y_test):
    """在测试集上评估模型，输出分类准确率"""
    accuracy = compute_accuracy(model, X_test, y_test)
    print(f"测试集上的分类准确率: {accuracy * 100:.2f}%")
    return accuracy


# CIFAR-10 类别标签
label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def visualize_predictions(model, X, y_true=None, num=16):
    """
    可视化模型对 CIFAR-10 输入的预测结果，展示为 4x4 网格
    """
    images = X[:num]
    preds = predict(model, images)

    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(4, 4, i + 1)

        # ✅ 正确还原图像格式
        img = images[i].reshape(32, 32, 3)       # (3072,) → (32, 32, 3)
        img = (img * 0.5) + 0.5                  # 还原 [-1, 1] → [0, 1]
        img = np.clip(img, 0, 1)                 # 避免像素溢出

        plt.imshow(img)
        pred_label = label_names[preds[i]]
        if y_true is not None:
            true_label = label_names[y_true[i]]
            plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=8)
        else:
            plt.title(f"Pred: {pred_label}", fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()

    def load_model(file_path):
        """
        从文件中加载模型参数
        :param file_path: 模型参数文件的路径
        :return: 加载好参数的模型
        """
        with open(file_path, 'rb') as f:
            model_params = pickle.load(f)

        # 假设已经有了模型的类定义，这里简单模拟创建一个模型实例
        # 你需要根据实际情况修改输入大小、隐藏层大小和输出大小
        input_size = 3 * 32 * 32
        hidden_size = 256
        output_size = 10
        model = ThreeLayerNet(input_size, hidden_size, output_size)

        # 将加载的参数赋值给模型
        model.params = model_params
        return model

    model = load_model(file_path='./best_model.pkl')
    acc = test_model(model, X_test, y_test)
    print(f'测试集准确率为： {acc}')

    # 展示前 10 张图片的预测结果
    visualize_predictions(model, X_test, y_test, num=16)




