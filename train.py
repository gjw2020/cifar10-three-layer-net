from  utils import *
from model import *

# 单轮训练
def train_one_epoch(model, X_train, y_train, batch_size, learning_rate):
    """
    执行一次epoch的训练
    :param model: ThreeLayerNet实例
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param batch_size: 每个小批量的样本数
    :param learning_rate: 当前学习率
    :return: 最后一个批次的损失
    """
    num_train = X_train.shape[0]
    X_train_shuffled, y_train_shuffled = shuffle_data(X_train, y_train)

    # 小批量迭代
    for i in range(0, num_train, batch_size):
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]
        # 计算当前批次的损失和梯度
        loss, grads = model.compute_loss_and_grads(X_batch, y_batch)
        # 参数更新（SGD）
        model.params['W1'] -= learning_rate * grads['W1']
        model.params['b1'] -= learning_rate * grads['b1']
        model.params['W2'] -= learning_rate * grads['W2']
        model.params['b2'] -= learning_rate * grads['b2']

    return loss


# 完整训练过程
def train_network(model, X_train, y_train, X_val, y_val,
                  epochs=20, batch_size=100, learning_rate=1e-3, lr_decay=0.95, patience=3):
    """
    使用小批量SGD训练给定的模型。
    model: ThreeLayerNet实例
    X_train, y_train: 训练数据及标签
    X_val, y_val: 验证集数据及标签
    epochs: 训练轮数
    batch_size: 每个小批量的样本数
    learning_rate: 初始学习率
    lr_decay: 学习率衰减因子（每个epoch乘以该值）
    patience: 早停等待轮数
    """
    best_val_acc = 0.0
    best_params = {}  # 用于存储最佳模型参数
    no_improvement_count = 0  # 记录验证集准确率没有提升的轮数
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, X_train, y_train, batch_size, learning_rate)
        train_losses.append(train_loss)

        # 计算验证集损失
        val_loss, _ = model.compute_loss_and_grads(X_val, y_val)
        val_losses.append(val_loss)

        # 每个epoch结束后评估训练和验证集准确率
        train_acc = compute_accuracy(model, X_train, y_train)
        val_acc = compute_accuracy(model, X_val, y_val)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 深拷贝当前参数
            best_params = {p: model.params[p].copy() for p in model.params}
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # 早停检查
        if no_improvement_count >= patience:
            print(f"早停: 在 {patience} 个epoch中验证集准确率没有提升。")
            break

        # 学习率衰减
        learning_rate *= lr_decay

    # 训练结束后，将模型参数设置为最佳参数
    model.params = best_params
    print(f"训练完成！最佳验证集准确率: {best_val_acc:.4f}")
    return model, train_losses, val_losses, val_accuracies