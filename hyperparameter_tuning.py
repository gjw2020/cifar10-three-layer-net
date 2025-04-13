from train import *
from utils import *
import os


def hyperparameter_tuning(X_train, y_train, X_val, y_val, input_size, output_size,
                          learning_rates=[1e-2, 1e-3, 1e-4],
                          hidden_sizes=[64, 128, 256],
                          reg_strengths=[0.0, 1e-3, 1e-4],
                          epochs=20, batch_size=128, lr_decay=0.95):
    best_model = None
    best_val_acc = 0.0
    best_hyperparams = {}

    all_train_losses = []
    all_val_losses = []
    all_val_accuracies = []
    all_param_combinations = []

    for lr in learning_rates:
        for h in hidden_sizes:
            for reg in reg_strengths:
                print(f"尝试超参数: lr={lr}, hidden_size={h}, reg={reg}")
                model = ThreeLayerNet(input_size=input_size, hidden_size=h, output_size=output_size,
                                      activation='relu', reg=reg)
                model, train_losses, val_losses, val_accuracies = train_network(model, X_train, y_train, X_val, y_val,
                                                                                epochs=epochs, batch_size=batch_size,
                                                                                learning_rate=lr, lr_decay=lr_decay)

                all_train_losses.append(train_losses)
                all_val_losses.append(val_losses)
                all_val_accuracies.append(val_accuracies)
                param_combination = f"lr={lr}, hs={h}, reg={reg}"
                all_param_combinations.append(param_combination)

                val_acc = compute_accuracy(model, X_val, y_val)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model
                    best_hyperparams = {'learning_rate': lr, 'hidden_size': h, 'reg': reg}
                    print(f"发现更好的超参数组合: {best_hyperparams}, 验证准确率 = {val_acc:.4f}")

    # 调用可视化函数
    visualize_training(all_train_losses, all_val_losses, all_val_accuracies, all_param_combinations)

    return best_model, best_hyperparams, best_val_acc