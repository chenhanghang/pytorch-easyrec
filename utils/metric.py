import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score, mean_squared_error


def accuracy(output, target): # 多分类准确率
    with torch.no_grad(): # 注意屏蔽梯度
        pred = torch.argmax(output, dim=1) # 返回最大值的下标
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1] # topk[1] 下标， top[0] 值
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

##### 排序的评价指标
def auc(output, target):
    with torch.no_grad():
        auc_score = metrics.roc_auc_score(target.cpu(), output.cpu())
    return auc_score

def get_metric_func(task_type="classification"):
    if task_type == "classification":
        return roc_auc_score
    elif task_type == "regression":
        return mean_squared_error
    else:
        raise ValueError("task_type must be classification or regression")