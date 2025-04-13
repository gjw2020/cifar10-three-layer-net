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

    model = load_model('https://drive.google.com/file/d/1GnPpTz2bGZZuR8_7qVd2iH6bYwFxocqv/view?usp=sharing')
    acc = test_model(model, X_test, y_test)
    print(f'测试集准确率为： {acc}')


