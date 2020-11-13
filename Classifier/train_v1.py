# -*- encoding: utf-8 -*-
'''
File    :   train_v0.py
Time    :   2020/11/09 17:44:34
Author  :   Chao Wang 
Version :   1.0
Contact :   374494067@qq.com
@Desc    :   None
'''



import time
import datetime
import pandas as pd
import numpy as np

import lightgbm as lgb
# # import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedKFold, KFold

# import catboost as cb

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


version = 'v1'
n_splits = 10


'''
data porcess
'''
def data_process():
    base = pd.read_csv('../data/train/base_info.csv')
    label = pd.read_csv('../data/train/entprise_info.csv')
    base = pd.merge(base, label, on=['id'], how='left') #基本信息 标签


    annual_report_info = pd.read_csv('../data/train/annual_report_info.csv')

    # annual_report_info['BUSSTNAME']=annual_report_info['BUSSTNAME'].fillna("无")
    annual_report_info['BUSSTNAME'] = annual_report_info['BUSSTNAME'].map({'无':-1,'开业':0, '歇业':1, '停业':2, '清算':3})

 

    # annual_report_info = annual_report_info.groupby('id',sort=False).agg('mean')
 
    # annual_report_info = pd.DataFrame(annual_report_info).reset_index()


    # annual_report_info = annual_report_info.drop(['ANCHEYEAR'],axis=1)
    
    # annual_report_info = annual_report_info.fillna(-1)


    # base = pd.merge(base, annual_report_info, on=['id'], how='left') #基本信息 标签



    annual_report_info = annual_report_info.loc[lambda x: x['ANCHEYEAR'] == 2018]

    annual_report_info_v1 = annual_report_info.drop_duplicates(keep=False)

    annual_report_info_v1 = annual_report_info_v1.sort_values('PUBSTATE',ascending=False)


    annual_report_info_v2_id = []
    annual_report_info_v2 = []

    for tup in annual_report_info_v1.itertuples():
        if tup[1] not in annual_report_info_v2_id:
            annual_report_info_v2.append(tup[1:])
            annual_report_info_v2_id.append(tup[1])


    annual_report_info_v2 = pd.DataFrame(annual_report_info_v2, columns=annual_report_info_v1.columns) 



    base = pd.merge(base, annual_report_info_v2, on=['id'], how='left') #基本信息 标签
    


    drop = ['opto','opform','WEBSITSIGN','FUNDAM','MEMNUM','FARNUM','ANNNEWMEMNUM','ANNREDMEMNUM']  
    

    for f in drop:
        del base[f]

    del base['dom'], base['opscope'] #单一值太多
    del base['oploc']

    #拆分年月特征
    base['year'] = base['opfrom'].apply(lambda x: int(x.split('-')[0]))
    base['month'] = base['opfrom'].apply(lambda x: int(x.split('-')[1]))
    del base['opfrom']

    data = base.copy()

    num_feat = []
    cate_feat = []

    drop = ['id', 'label'] #不需要的特征
    cat = ['industryphy'] #类别特征
    for j in list(data.columns): 
        if j in drop:
            continue
        if j in cat:
            cate_feat.append(j)
        else:
            num_feat.append(j)
            
    for i in cate_feat:
        data[i] = data[i].astype('category')
    features = num_feat + cate_feat
    print(features)

    # for f in num_feat:
    #     data[f].fillna(0,inplace=True)

    return data,features,cate_feat

def get_predict_w(model, data, label='label', feature=[], cate_feature=[], random_state=2018, n_splits=5,
                  model_type='lgb'):
    feature_importance = {}

    if 'sample_weight' not in data.keys():
        data['sample_weight'] = 1
    model.random_state = random_state
    predict_label = 'predict_' + label
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    data[predict_label] = 0
    test_index = (data[label].isnull()) | (data[label] == -1)    #找到要预测的数据集
    train_data = data[~test_index].reset_index(drop=True)     #分割出预测集训练集
    test_data = data[test_index]

    for train_idx, val_idx in kfold.split(train_data):
        model.random_state = model.random_state + 1

        train_x = train_data.loc[train_idx][feature]
        train_y = train_data.loc[train_idx][label]

        test_x = train_data.loc[val_idx][feature]
        test_y = train_data.loc[val_idx][label]
        if model_type == 'lgb':
            try:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=400,
                          eval_metric='mae',
                          callbacks=[lgb.reset_parameter(learning_rate=lambda iter: max(0.005, 0.5 * (0.99 ** iter)))],
                          categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100)
            except:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=200,
                          eval_metric='mae',
                          callbacks=[lgb.reset_parameter(learning_rate=lambda iter: max(0.005, 0.5 * (0.99 ** iter)))],
                          categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100)

            importance_df = pd.DataFrame()
            importance_df["feature"] = train_x.columns.tolist()      
            importance_df["importance"] = model.booster_.feature_importance('gain')
            
            for f,importance in zip(train_x.columns.tolist(), model.booster_.feature_importance('gain')):
                if f not in feature_importance:
                    feature_importance[f] = importance
                else:
                    feature_importance[f] += importance


        elif model_type == 'ctb':
            model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=200,
                      # eval_metric='mae',
                      # callbacks=[lgb.reset_parameter(learning_rate=lambda iter: max(0.005, 0.5 * (0.99 ** iter)))],
                      cat_features=cate_feature,
                      sample_weight=train_data.loc[train_idx]['sample_weight'],
                      verbose=100)
        train_data.loc[val_idx, predict_label] = model.predict(test_x)
        if len(test_data) != 0:                  #预测集的预测
            test_data[predict_label] = test_data[predict_label] + model.predict(test_data[feature])
    test_data[predict_label] = test_data[predict_label] / n_splits
    # print((train_data[label], train_data[predict_label]) * 5, train_data[predict_label].mean(),
    #       test_data[predict_label].mean())
    # print('########################################')


    return pd.concat([train_data, test_data], sort=True, ignore_index=True), predict_label,feature_importance

def display_feature_importance(feature_importance):
    f_list = []
    importance_list = []
    for k,v in feature_importance.items():
        f_list.append(k)
        importance_list.append(v)

    feature_importance = {'feature':f_list,'importance':importance_list}
    feature_importance = pd.DataFrame(feature_importance, columns=['feature','importance']) 
    feature_importance["importance"] = feature_importance["importance"].apply(lambda x:x/n_splits)
    feature_importance = feature_importance.sort_values(by="importance", ascending=False)
    print('###################################')
    print('前20的特征：')
    print(feature_importance[:20])
    print('###################################')

    print('\n')
    print('###################################')
    print('无用的特征：')
    df_2_neglect = feature_importance[feature_importance['importance'] < 1e-1]
    print('###################################')
    print(df_2_neglect)
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", data= feature_importance[:20])
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('polt/lgbm_importances_'+str(version)+'.png')


if __name__ == "__main__":


    data,features,cate_feat = data_process()
    # cb_model = cb.CatBoostRegressor()

    lgb_model = lgb.LGBMRegressor(
        num_leaves=64, reg_alpha=0., reg_lambda=0.01, metric='rmse',
        max_depth=-1, learning_rate=0.05, min_child_samples=10, seed=202011,
        n_estimators=2000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    )

    data, predict_label,feature_importance = get_predict_w(lgb_model, data, label='label',
                                        feature=features, cate_feature=cate_feat,
                                        random_state=202011, n_splits=n_splits,  model_type='lgb')

    display_feature_importance(feature_importance)
    data['score'] = data[predict_label]
    #data['forecastVolum'] = data['lgb'].apply(lambda x: -x if x < 0 else x)
    df = data[data.label.isnull()][['id', 'score']]
    df['score'] = df['score'].apply(lambda x: 0 if x<0 else x) #修正
    df['score'] = df['score'].apply(lambda x: 1 if x>1 else x)
    df.to_csv('../result/11_10/submit_'+str(version)+'.csv', index=False) #submit


