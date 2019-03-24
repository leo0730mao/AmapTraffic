import pickle
import numpy as np
import xgboost as xgb


def rmspe_xg(yhat, y):
    # y DMatrix对象
    y = y.get_label()
    # y.get_label 二维数组
    y = np.exp(y)  # 二维数组
    yhat = np.exp(yhat)  # 一维数组
    rmspe = np.sqrt(np.mean((y - yhat) ** 2))
    return "rmspe", rmspe


def train(dtrain, watchlist):
    num_trees = 450
    # num_trees=45
    params = {"eta": 0.15,
              "max_depth": 8,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "silent": 1
              }

    # 训练模型
    gbm = xgb.train(params, dtrain, num_trees, early_stopping_rounds = 50, evals = watchlist, verbose_eval = True)


def prepare_data(path, mode):
    with open("%s/%s_purposed.dat" % (path, mode), 'rb') as f:
        data = pickle.load(f)
    y = data['y'].flatten()
    data = data['X']
    for i in range(len(data)):
        data[i] = data[i].flatten()
    x = None
    for d in data:
        if x is None:
            x = d
        else:
            x = np.column_stack((x, d))
    res = xgb.DMatrix(x, label = y)
    return res


if __name__ == '__main__':
    dtrain = prepare_data("F:/DATA/dataset/v1", 'train')
    dvalid = prepare_data("F:/DATA/dataset/v1", 'test')
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    train(dtrain, watchlist)
