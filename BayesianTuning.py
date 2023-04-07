# -*- coding: utf-8 -*-
'''
贝叶斯调参
Created on 2022年1月23日
@author: xiaoyw71
'''

import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class BayesianTuningClassi(object):
    def __init__(self, modelname, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.modelname = modelname  # 算法模型名称

    def getModelFun(self):

        def _RF_evaluate(n_estimators, min_samples_split, max_features, max_depth):
            from sklearn.ensemble import RandomForestClassifier

            val = cross_val_score(
                RandomForestClassifier(n_estimators=int(n_estimators),
                                       min_samples_split=int(
                                           min_samples_split),
                                       max_features=min(max_features, 0.999),
                                       max_depth=int(max_depth),
                                       random_state=2,
                                       n_jobs=-1),
                self.x_train, self.y_train, scoring='f1', cv=3
            ).mean()

            return val

        def initXgbDTrain():
            import xgboost as xgb
            return xgb.DMatrix(self.x_train, label=self.y_train)

        def _xgb_multi_evaluate(max_depth, gamma, colsample_bytree, subsample, min_child_weight):
            import xgboost as xgb

            params = {
                'objective': 'multi:softprob',  # 多分类的问题
                'num_class': 2,                 # 类别数为2，与 multisoftmax 并用
                'eval_metric': 'mlogloss',
                'max_depth': int(max_depth),
                'subsample': subsample,  # 0.8
                'eta': 0.5,
                'gamma': gamma,
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight}

            cv_result = xgb.cv(params, self.dtrain,
                               num_boost_round=20, nfold=3)

            return -1.0 * cv_result['test-mlogloss-mean'].iloc[-1]

        def _xgb_logistic_evaluate(max_depth, subsample, gamma, colsample_bytree, min_child_weight):
            import xgboost as xgb

            params = {
                'objective': 'binary:logistic',  # 逻辑回归二分类的问题
                'eval_metric': 'auc',
                'max_depth': int(max_depth),
                'subsample': subsample,  # 0.8
                'eta': 0.2,
                'gamma': gamma,
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight}

            # nfold 分三组交叉验证 ，曾经用到5
            cv_result = xgb.cv(params, self.dtrain,
                               num_boost_round=30, nfold=5)

            return 1.0 * cv_result['test-auc-mean'].iloc[-1]

        if self.modelname == 'RF':
            bo_f = _RF_evaluate
        elif self.modelname == 'XGB_multi':
            self.dtrain = initXgbDTrain()
            bo_f = _xgb_multi_evaluate
        elif self.modelname == 'XGB_logistic':
            self.dtrain = initXgbDTrain()
            bo_f = _xgb_logistic_evaluate
        elif self.modelname == 'BPNet':
            pass

        return bo_f

    def evaluate(self, bo_f, pbounds, init_points, n_iter):

        bo = BayesianOptimization(
            f=bo_f,   # 目标函数
            pbounds=pbounds,  # 取值空间
            verbose=2,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
            random_state=1,
        )

        bo.maximize(init_points=init_points,   # 随机搜索的步数
                    n_iter=n_iter,       # 执行贝叶斯优化迭代次数
                    acq='ei')

        print(bo.max)
        res = bo.max
        params_max = res['params']

        return params_max

# 从文件获取训练集


def getTrainDatasByFile(dfile, flag_col='flag'):
    df = pd.read_csv(dfile)

    data = df.drop([flag_col], axis=1)
    Y = df[flag_col]

    return data, Y

# 从数据库获取训练集


def getTrainDatasByDB(model_name='XGB_multi', feature_name='S60', flag_col='flag'):
    from PredictionModel.ChurnTrainCollectionFromMongo import getChurnTrainStatusCollection2
    from PredictionModel.CustomerFeatureUtils import DB_info, Churn_Feature, getLogFeature

    coll_name = DB_info.ChurnTrainCollection2[feature_name]
    df = getChurnTrainStatusCollection2(model_name, coll_name, feature_name)
    df = df.loc[df['split'] < 1]
    print('Train Data Counter is ', len(df))
    Y = df[[flag_col]]

    # cols = DB_info.Churn_Feature2L.copy()
    cols = Churn_Feature(model_name, feature_name)
    cols.remove('carduser_id')
    cols.remove('split')
    cols.remove(flag_col)
    df = df[cols]
    df.fillna(0, inplace=True)
    #df = getLogFeature(df, cols, balance=True)
    #df.loc[:, 'balance'] = np.log(df['balance'] + 1)
    print(df.columns)

    return df, Y


# 特征归一化
def feature_StandardScaler(data, normalization=True):
    # 数据归一化处理
    if normalization:
        scaler = StandardScaler()
        columns = data.columns
        indexs_train = data.index
        data = pd.DataFrame(scaler.fit_transform(
            data), index=indexs_train, columns=columns)

    return data


# 随机森林模型调参
def RF_evaluate(feature_name='S0'):
    from sklearn.model_selection import train_test_split

    # 确定取值空间
    pbounds = {'n_estimators': (10, 200),  # 表示取值范围为10至250
               'min_samples_split': (2, 25),
               'max_features': (0.1, 0.999),
               'max_depth': (3, 4)}

    dfile = 'datas1.csv'
    modelname = 'RF'

    # data, Y = getTrainDatasByFile(dfile)
    data, Y = getTrainDatasByDB(model_name='RF', feature_name=feature_name)
    # 查找空问题
    #kk = data[data.isnull().any(axis=1)==True]
    #kk2 = data[data.isna().any(axis=1)==True]
    # print(kk)
    # print(kk2)

    # 数据归一化处理
    X = feature_StandardScaler(data)
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    x_train = X
    y_train = Y.values.ravel()

    BTC = BayesianTuningClassi(modelname, x_train, y_train)
    bo_f = BTC.getModelFun()
    BTC.evaluate(bo_f, pbounds, 5, 25)

# XGB模型调参


def XGB_multi_evaluate(feature_name='S0'):
    from sklearn.model_selection import train_test_split
    pbounds = {'max_depth': (2, 4),
               'gamma': (0.1, 1.0),
               'colsample_bytree': (0.5, 0.95),
               'subsample': (0.50, 0.95),
               'min_child_weight': (10, 300)}
    dfile = 'datas0.csv'
    modelname = 'XGB_multi'

    #data , Y = getTrainDatasByFile(dfile)
    data, Y = getTrainDatasByDB(
        model_name='XGB_multi', feature_name=feature_name)

    # 数据归一化处理
    X = feature_StandardScaler(data)
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    x_train = X
    y_train = Y

    BTC = BayesianTuningClassi(modelname, x_train, y_train)
    bo_f = BTC.getModelFun()
    BTC.evaluate(bo_f, pbounds, 5, 25)


def XGB_logistic_evaluate(feature_name='S0'):
    from sklearn.model_selection import train_test_split
    pbounds = {'max_depth': (6, 10),
               'gamma': (0.1, 1.0),
               'colsample_bytree': (0.6, 0.95),
               'subsample': (0.60, 0.95),
               'min_child_weight': (50, 800)}

    dfile = 'datas0.csv'
    modelname = 'XGB_logistic'

    #data , Y = getTrainDatasByFile(dfile)
    data, Y = getTrainDatasByDB(
        model_name='XGB_logistic', feature_name=feature_name)

    # 数据归一化处理
    X = feature_StandardScaler(data)
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    x_train = X
    y_train = Y

    BTC = BayesianTuningClassi(modelname, x_train, y_train)
    bo_f = BTC.getModelFun()
    BTC.evaluate(bo_f, pbounds, 5, 30)


if __name__ == '__main__':
    splitname = ['S0', 'S30', 'S60', 'S90']    # 分阶段名称
    print('请输入数字选择训练集：')
    print('0, 小于30天活跃客户')
    print('1, 大于等于30天，小于60天活跃客户')
    print('2, 大于等于60天，小于90天活跃客户')
    print('3, 大于等于90天，小于180天活跃客户')

    select = input('请输入选择数字 ： ')
    select = int(select)
    if select not in [0, 1, 2, 3]:
        print('输入数字超限')
        sys.exit(0)
    else:

        print('请输入数字选择基础模型：')
        print('0, XGB多分类')
        print('1, XGB逻辑回归')
        print('2, 随机森林')
        print('3, 神经网络')
        model = input('请输入选择数字 ： ')
        model = int(model)
        if model not in [0, 1, 2, 3]:
            print('输入数字超限')
            sys.exit(0)

        elif model == 0:
            XGB_multi_evaluate(splitname[select])

        elif model == 1:
            XGB_logistic_evaluate(splitname[select])

        elif model == 2:
            RF_evaluate(splitname[select])

        elif model == 3:
            pass

    exit()
