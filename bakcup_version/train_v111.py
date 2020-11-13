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

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# import catboost as cb

'''
data porcess
'''
no_important = []

def data_process():
    base = pd.read_csv('./data/train/base_info.csv')
    label = pd.read_csv('./data/train/entprise_info.csv')
    base = pd.merge(base, label, on=['id'], how='left') #基本信息 标签


    annual_report_info = pd.read_csv('./data/train/annual_report_info.csv')
    annual_report_info['BUSSTNAME'] = annual_report_info['BUSSTNAME'].map({'开业': 0,'歇业': 1,'停业': 1})



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
    

    # '''
    # add news_info
    # '''
    # news_info = pd.read_csv('./data/train/news_info.csv')
    # news_score = {}
    # news_map_score = {
    #     '消极':-1,
    #     '中立':0,
    #     '积极':1
    # }
    # news_score_id_list = []
    # news_score_score_list = []
    # for tup in news_info.itertuples():
    #     if tup[1] not in news_score: 
    #         news_score[tup[1]] = news_map_score[tup[2]]
    #     else:
    #         news_score[tup[1]] += news_map_score[tup[2]]
    # for k,v in news_score.items():
    #     news_score_id_list.append(k)
    #     news_score_score_list.append(v)

    # news_score_dict = {'id':news_score_id_list,'news_score':news_score_score_list}
    # news_score_df = pd.DataFrame(news_score_dict, columns=['id','news_score'])
    # base = pd.merge(base, news_score_df, on=['id'], how='left') #基本信息 标签
    # base['news_score'].fillna(0,inplace=True)


    # '''
    # add change_info
    # '''
    # change_info = pd.read_csv('./data/train/change_info.csv')
    # change_counts = {}

    # change_counts_id_list = []
    # change_counts_list = []
    # for tup in change_info.itertuples():
    #     if tup[1] not in change_counts: 
    #         change_counts[tup[1]] = 1
    #     else:
    #         change_counts[tup[1]] += 1
    # for k,v in change_counts.items():
    #     change_counts_id_list.append(k)
    #     change_counts_list.append(v)

    # change_counts_dict = {'id':change_counts_id_list,'change_counts':change_counts_list}
    # change_counts_df = pd.DataFrame(change_counts_dict, columns=['id','change_counts'])

    # base = pd.merge(base, change_counts_df, on=['id'], how='left') #基本信息 标签
    # base['change_counts'].fillna(0,inplace=True)
    # # print(base)


    # # #缺失值太多  0.80630422954 
    # # drop = ['enttypeitem', 'opto', 'empnum', 'compform', 'parnum',
    # #     'exenum', 'opform', 'ptbusscope', 'venind', 'enttypeminu',
    # #     'midpreindcode', 'protype', 'reccap', 'forreccap',
    # #     'forregcap', 'congro']

    # # drop = ['opto','opform']  #0.81445714244 , 0.81514767439

    # # drop = ['opto','opform','WEBSITSIGN','FUNDAM','MEMNUM','FARNUM','ANNNEWMEMNUM','ANNREDMEMNUM']  
    
    # # drop = []

    # # drop1 = ['BUSSTNAME', 'DISEMPLNUM', 'DISPERNUM', 'RETEMPLNUM', 'RETSOLNUM', 'STATE', 'UNEEMPLNUM', 'adbusign', 'compform', 'congro', 'forreccap', 'forregcap', 'industryco', 'midpreindcode', 'protype', 'ptbusscope', 'venind', 'ANCHEYEAR', 'STOCKTRANSIGN', 'state', 'EMPNUMSIGN', 'UNENUM', 'enttype', 'regtype']
    # drop = ['ANNNEWMEMNUM', 'ANNREDMEMNUM', 'DISEMPLNUM', 'DISPERNUM', 'EMPNUMSIGN', 'FARNUM', 'FUNDAM', 'MEMNUM', 'RETEMPLNUM', 'RETSOLNUM', 'STATE', 'UNEEMPLNUM', 'UNENUM', 'adbusign', 'compform', 'congro', 'enttype', 'forreccap', 'forregcap', 'midpreindcode', 'opform', 'protype', 'ptbusscope', 'state', 'venind', 'WEBSITSIGN', 'ANCHEYEAR', 'STOCKTRANSIGN', 'regtype','FORINVESTSIGN', 'regcap']
    # drop += ['exenum', 'reccap', 'BUSSTNAME']
    # for f in drop:
    #     del base[f]



    # #拆分年月特征
    # base['from_year'] = base['opfrom'].apply(lambda x: int(x.split('-')[0]))
    # base['from_month'] = base['opfrom'].apply(lambda x: int(x.split('-')[1]))
  
    # del base['dom']
    # del base['opscope'] #单一值太多
    # del base['oploc']
    # del base['opfrom']
    # del base['opto']

    # # base['opform'] = base['opform'].astype('category')


    # data = base.copy()

    # num_feat = []
    # cate_feat = []

    # drop = ['id', 'label'] #不需要的特征

    # # cat = ['industryphy'] #类别特征
    # cat = ['industryphy','enttype'] #类别特征

    # for j in list(data.columns): 
    #     if j in drop:
    #         continue
    #     if j in cat:
    #         cate_feat.append(j)
    #     else:
    #         num_feat.append(j)
            
    # for i in cate_feat:
    #     data[i] = data[i].astype('category')
    # features = num_feat + cate_feat
    # print(features)

    # # for f in num_feat:
    # #     data[f].fillna(0,inplace=True)

    drop = ['opto','opform','WEBSITSIGN','FUNDAM','MEMNUM','FARNUM','ANNNEWMEMNUM','ANNREDMEMNUM']  
    drop += ['BUSSTNAME', 'DISEMPLNUM', 'DISPERNUM', 'RETEMPLNUM', 'RETSOLNUM', 'STATE', 'UNEEMPLNUM', 'adbusign', 'compform', 'congro', 'forreccap', 'forregcap', 'industryco', 'midpreindcode', 'protype', 'ptbusscope', 'venind', 'ANCHEYEAR', 'STOCKTRANSIGN', 'state', 'EMPNUMSIGN', 'UNENUM', 'enttype', 'regtype']

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
    return data,features,cate_feat



def display_importances(feature_importance_df_, doWorst=False, n_feat=50):
    # Plot feature importances
    if not doWorst:
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:n_feat].index        
    else:
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[-n_feat:].index
    
    mean_imp = feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    df_2_neglect = mean_imp[mean_imp['importance'] < 1e-2]

    # print('The list of features with 0 importance: ')
    # print(df_2_neglect.index.values.tolist())

    for item in df_2_neglect.index.values.tolist():
        if item not in no_important:
            no_important.append(item)


    del mean_imp, df_2_neglect
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


def get_predict_w(model, data, label='label', feature=[], cate_feature=[], random_state=2018, n_splits=5,
                  model_type='lgb'):
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
            display_importances(feature_importance_df_=importance_df, n_feat=20)

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


    return pd.concat([train_data, test_data], sort=True, ignore_index=True), predict_label


if __name__ == "__main__":


    data,features,cate_feat = data_process()
    # cb_model = cb.CatBoostRegressor()

    lgb_model = lgb.LGBMRegressor(
        num_leaves=64, reg_alpha=0., reg_lambda=0.01, metric='rmse',
        max_depth=-1, learning_rate=0.05, min_child_samples=10, seed=202011,
        n_estimators=2000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    )

    data, predict_label = get_predict_w(lgb_model, data, label='label',
                                        feature=features, cate_feature=cate_feat,
                                        random_state=202011, n_splits=10,  model_type='lgb')

    data['score'] = data[predict_label]
    #data['forecastVolum'] = data['lgb'].apply(lambda x: -x if x < 0 else x)
    df = data[data.label.isnull()][['id', 'score']]
    df['score'] = df['score'].apply(lambda x: 0 if x<0 else x) #修正
    df['score'] = df['score'].apply(lambda x: 1 if x>1 else x)
    df.to_csv('result/submit_v111.csv', index=False) #submit

    print(no_important)

