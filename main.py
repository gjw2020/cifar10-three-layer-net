from data_load import *
from hyperparameter_tuning import *
from model import ThreeLayerNet
from predict import *
from train import *
from utils import *


def main():
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()

    # 模型训练相关参数
    input_size = 3 * 32 * 32
    output_size = 10

    # 超参数调优
    best_model, best_hyperparams, best_val_acc = hyperparameter_tuning(X_train, y_train, X_val, y_val, input_size, output_size,
                          learning_rates=[1e-2, 1e-3, 1e-4],
                          hidden_sizes=[64, 128, 256],
                          reg_strengths=[0.0, 1e-3, 1e-4],
                          epochs=20, batch_size=128, lr_decay=1)

    # 保存最佳模型
    save_model(best_model, 'best_model.pkl')

    # 加载模型
    loaded_model = ThreeLayerNet(input_size, best_hyperparams['hidden_size'], output_size, activation='relu', reg=best_hyperparams['reg'])
    loaded_model = load_model(loaded_model, 'best_model.pkl')

    # 在测试集上评估模型
    test_model(loaded_model, X_test, y_test)

    # 可视化模型参数并保存图片
    visualize_model_parameters(loaded_model, save_path='model_params_vis.png')

if __name__ == "__main__":
    main()