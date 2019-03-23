import xgboost as xgb

num_trees = 450
# num_trees=45
params = {"objective": "reg:squarederror",
          "eta": 0.15,
          "max_depth": 8,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1
          }

# 训练模型
gbm = xgb.train(params, dtrain, num_trees, evals = watchlist, early_stopping_rounds = 50, feval = rmspe_xg, verbose_eval = True)