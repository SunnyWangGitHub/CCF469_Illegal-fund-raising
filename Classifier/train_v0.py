# -*- encoding: utf-8 -*-
'''
File    :   train_v0py
Time    :   2020/11/10 16:19:17
Author  :   Chao Wang 
Version :   1.0
Contact :   374494067@qq.com
@Desc    :   None
'''

import pandas as pd
from tqdm import *
from sklearn.metrics import f1_score,precision_recall_fscore_support,roc_curve,auc,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import catboost as cab
import lightgbm as lgb
import numpy as np

version = 'v0'
# n_splits = 10

def data_process():
    base_info=pd.read_csv('../data/train/base_info.csv')#企业的基本信息
    annual_report_info=pd.read_csv('../data/train/annual_report_info.csv')#企业的年报基本信息
    tax_info=pd.read_csv('../data/train/tax_info.csv')#企业的纳税信息
    change_info=pd.read_csv('../data/train/tax_info.csv')#变更信息
    news_info=pd.read_csv('../data/train/news_info.csv')#舆情信息
    other_info=pd.read_csv('../data/train/other_info.csv')#其它信息
    entprise_info=pd.read_csv('../data/train/entprise_info.csv')#企业标注信息{0: 13884, 1: 981}
    entprise_evaluate=pd.read_csv('../data/entprise_evaluate.csv')#未标注信息


    # print('base_info shape:',base_info.shape,'id unique:',len(base_info['id'].unique()))
    # print('annual_report_info shape:',annual_report_info.shape,'id unique:',len(annual_report_info['id'].unique()))
    # print('tax_info shape:',tax_info.shape,'id unique:',len(tax_info['id'].unique()))
    # print('change_info shape:',change_info.shape,'id unique:',len(change_info['id'].unique()))
    # print('news_info shape:',news_info.shape,'id unique:',len(news_info['id'].unique()))
    # print('other_info shape:',other_info.shape,'id unique:',len(other_info['id'].unique()))
    # print('entprise_info shape:',entprise_info.shape,'id unique:',len(entprise_info['id'].unique()))
    # print('entprise_evaluate shape:',entprise_evaluate.shape,'id unique:',len(entprise_evaluate['id'].unique()))



    #下面看一下具有企业年报信息和纳税信息的企业有多少是非法集资的企业
    #首先筛选出非法集资的企业
    # illegal_id_list=[]
    # legal_id_list=[]
    # for index,name_id,flag in entprise_info.itertuples():
    #     if flag==1:
    #         illegal_id_list.append(name_id)
    #     else:
    #         legal_id_list.append(name_id)
    # print(len(legal_id_list),len(illegal_id_list),len(legal_id_list)/len(illegal_id_list))


    # #..................年报基本信息信息数据...................
    # cnt_list_annual={'-1':0,'0':0,'1':0}
    # for i in annual_report_info['id'].unique():
    #     if i in illegal_id_list:
    #         cnt_list_annual['1']+=1
    #     elif i in legal_id_list:
    #         cnt_list_annual['0']+=1
    #     else:
    #         cnt_list_annual['-1']+=1
    # #具有年报基本信息的企业中，有536违法；2800合法；5601为测试集
    # print("具有年报基本信息的企业中，有{}违法；{}合法；{}为测试集".format(cnt_list_annual['1'],cnt_list_annual['0'],cnt_list_annual['-1']))
    # #合法/违法:5.223880597014926，说明具有年报信息的企业，是非法的概率很高 由此可见，年报信息很重要，这是十分重要的特征
    # print("具有年报基本信息的企业中：合法/违法:{}".format(cnt_list_annual['0']/cnt_list_annual['1']))
    # print("不具有年报基本信息的企业中：合法/违法:{}".format((len(legal_id_list)-cnt_list_annual['0'])/(len(illegal_id_list)-cnt_list_annual['1'])))

    # print('\n')
    # #由此可见，纳税信息很重要，这是十分重要的特征
    # #...........................纳税信息news_info....................
    # cnt_list_annual={'-1':0,'0':0,'1':0}
    # for i in tax_info['id'].unique():
    #     if i in illegal_id_list:
    #         cnt_list_annual['1']+=1
    #     elif i in legal_id_list:
    #         cnt_list_annual['0']+=1
    #     else:
    #         cnt_list_annual['-1']+=1
    # #具有年报基本信息的企业中，有536违法；2800合法；5601为测试集
    # print("具有纳税基本信息的企业中，有{}违法；{}合法；{}为测试集".format(cnt_list_annual['1'],cnt_list_annual['0'],cnt_list_annual['-1']))
    # #合法/违法:5.223880597014926，说明具有年报信息的企业，是非法的概率很高 由此可见，年报信息很重要，这是十分重要的特征
    # print("具有纳税信息的企业中：合法/违法:{}".format(cnt_list_annual['0']/cnt_list_annual['1']))
    # print("不具纳税信息的企业中：合法/违法:{}".format((len(legal_id_list)-cnt_list_annual['0'])/(len(illegal_id_list)-cnt_list_annual['1'])))

    #空值大于0.5的列都删除掉
    annual_report_info_clean=annual_report_info.dropna(thresh=annual_report_info.shape[0]*0.5,how='all',axis=1)
    
    annual_report_info['BUSSTNAME']=annual_report_info['BUSSTNAME'].fillna("无")
    annual_report_info['BUSSTNAME'] = annual_report_info['BUSSTNAME'].map({'无':-1,'开业':0, '歇业':1, '停业':2, '清算':3})
    annual_report_info = annual_report_info.groupby('id',sort=False).agg('mean')
    annual_report_info = pd.DataFrame(annual_report_info).reset_index()
    annual_report_info = annual_report_info.drop(['ANCHEYEAR'],axis=1)
    annual_report_info = annual_report_info.fillna(-1)

    # annual_report_info = annual_report_info.loc[lambda x: x['ANCHEYEAR'] == 2018]
    # annual_report_info_v1 = annual_report_info.drop_duplicates(keep=False)
    # annual_report_info_v1 = annual_report_info_v1.sort_values('PUBSTATE',ascending=False)


    # annual_report_info_v2_id = []
    # annual_report_info_v2 = []

    # for tup in annual_report_info_v1.itertuples():
    #     if tup[1] not in annual_report_info_v2_id:
    #         annual_report_info_v2.append(tup[1:])
    #         annual_report_info_v2_id.append(tup[1])


    # annual_report_info = pd.DataFrame(annual_report_info_v2, columns=annual_report_info_v1.columns) 




    #处理tax数据
    tax_info_clean = tax_info.drop(['START_DATE','END_DATE'],axis=1)
    tax_info_clean['TAX_CATEGORIES'] = tax_info_clean['TAX_CATEGORIES'].fillna("无")
    tax_info_clean['TAX_ITEMS'] = tax_info_clean['TAX_ITEMS'].fillna("无")

    TAX_CATEGORIES = tax_info['TAX_CATEGORIES'].unique()
    TAX_CATEGORIES_map = {}
    for i,c in enumerate(TAX_CATEGORIES):
        TAX_CATEGORIES_map[c] = i

    tax_info_clean['TAX_CATEGORIES'] = tax_info_clean['TAX_CATEGORIES'].map(TAX_CATEGORIES_map)


    TAX_ITEMS = tax_info['TAX_ITEMS'].unique()
    TAX_ITEMS_map = {}
    for i,c in enumerate(TAX_ITEMS):
        TAX_ITEMS_map[c] = i

    tax_info_clean['TAX_ITEMS'] = tax_info_clean['TAX_ITEMS'].map(TAX_ITEMS_map)
    tax_info_clean = tax_info_clean.fillna(-1)
    #
    tax_info_clean_group = tax_info_clean.groupby('id',sort=False).agg('mean')
    
    tax_info_clean = pd.DataFrame(tax_info_clean_group).reset_index()



    # #处理base_info数据
    base_info_clean = base_info.drop(['opscope','opfrom','opto','dom'],axis=1)

    #............................对object类型进行编码...............................
    base_info_clean['industryphy']=base_info_clean['industryphy'].fillna("无")
    # base_info_clean['dom']=base_info_clean['dom'].fillna("无")
    base_info_clean['opform']=base_info_clean['opform'].fillna("无")
    base_info_clean['oploc']=base_info_clean['oploc'].fillna("无")



    Industryphy = base_info_clean['industryphy'].unique()
    Industryphy_map = {}
    for i,c in enumerate(Industryphy):
        Industryphy_map[c] = i

    base_info_clean['industryphy'] = base_info_clean['industryphy'].map(Industryphy_map)



    Opform = base_info_clean['opform'].unique()
    Opform_map = {}
    for i,c in enumerate(Opform):
        Opform_map[c] = i

    base_info_clean['opform'] = base_info_clean['opform'].map(Opform_map)


    Oploc = base_info_clean['oploc'].unique()
    Oploc_map = {}
    for i,c in enumerate(Oploc):
        Oploc_map[c] = i

    base_info_clean['oploc'] = base_info_clean['oploc'].map(Oploc_map)


    base_info_clean = base_info_clean.fillna(-1)



    #........................分桶.................................
    def bucket(name,bucket_len):
        gap_list=[base_info_clean[name].quantile(i/bucket_len) for i in range(bucket_len+1)]
        len_data=len(base_info_clean[name])
        new_col=[]
        for i in base_info_clean[name].values:
            for j in range(len(gap_list)):
                if gap_list[j]>=i:
                    encode=j
                    break
            new_col.append(encode)
        return new_col

    #注册资本_实缴资本
    base_info_clean['regcap_reccap'] = base_info_clean['regcap']- base_info_clean['reccap']
    # 注册资本分桶
    base_info_clean['regcap'] = base_info_clean['regcap'].fillna(base_info_clean['regcap'].median())
    base_info_clean['bucket_regcap'] = bucket('regcap',5)
    #实缴资本分桶
    base_info_clean['reccap']=base_info_clean['reccap'].fillna(base_info_clean['reccap'].median())
    base_info_clean['bucket_reccap'] = bucket('reccap',5)
    # 注册资本_实缴资本分桶
    base_info_clean['regcap_reccap'] = base_info_clean['regcap_reccap'].fillna(base_info_clean['regcap_reccap'].median())
    base_info_clean['bucket_regcap_reccap'] = bucket('regcap_reccap',5)
    print('分桶完毕.................')



    #.............................交叉.........................
    #作两个特征的交叉
    def cross_two(name_1,name_2):
        new_col=[]
        encode=0
        dic={}
        val_1=base_info[name_1]
        val_2=base_info[name_2]
        for i in tqdm(range(len(val_1))):
            tmp=str(val_1[i])+'_'+str(val_2[i])
            if tmp in dic:
                new_col.append(dic[tmp])
            else:
                dic[tmp]=encode
                new_col.append(encode)
                encode+=1
        return new_col

    def cross_two_2(df,name_1,name_2):
        new_col=[]
        encode=0
        dic={}
        val_1=df[name_1]
        val_2=df[name_2]
        for i in tqdm(range(len(val_1))):
            tmp=str(val_1[i])+'_'+str(val_2[i])
            if tmp in dic:
                new_col.append(dic[tmp])
            else:
                dic[tmp]=encode
                new_col.append(encode)
                encode+=1
        return new_col

    #企业类型-小类的交叉特征
    base_info_clean['enttypegb']=base_info_clean['enttypegb'].fillna("无")
    base_info_clean['enttypeitem']=base_info_clean['enttypeitem'].fillna("无")
    new_col = cross_two('enttypegb','enttypeitem')#作企业类型-小类的交叉特征
    base_info_clean['enttypegb_enttypeitem']=new_col

    #企业类型-细类的交叉特征
    # base_info_clean['enttypeminu']= base_info_clean['enttypeminu'].fillna("无")
    # new_col = cross_two('enttypegb','enttypeminu')#作企业类型-小类的交叉特征
    # base_info_clean['enttypegb_enttypeminu']=new_col

    #
    # 行业类别-细类的交叉特征
    base_info_clean['industryphy']= base_info_clean['industryphy'].fillna("无")
    base_info_clean['industryco']= base_info_clean['industryco'].fillna("无")
    new_col = cross_two('industryphy','industryco')#作企业类型-小类的交叉特征
    base_info_clean['industryphy_industryco']= new_col

    # base_info_clean['regcap'] = base_info_clean['regcap'].fillna(base_info_clean['regcap'].median())
    # base_info_clean['reccap']= base_info_clean['reccap'].fillna(base_info_clean['reccap'].median())
    # new_col = cross_two('regcap','reccap')#作企业类型-小类的交叉特征
    # base_info_clean['regcap_reccap_cross']= new_col   




    print('交叉特征完毕.................')

    # print(base_info_clean['venind'])


    # cat_features = ['opform','oploc',
    #             'enttypegb','enttypeitem','enttypegb_enttypeitem','industryco','industryphy_industryco',
    #             'adbusign','townsign','regtype','TAX_CATEGORIES'
    #             ]


    cat_features=['industryphy','opform','oploc','bucket_regcap',
                'bucket_reccap','bucket_regcap_reccap',
                'enttypegb','enttypeitem','enttypegb_enttypeitem',
                'industryphy','industryco','industryphy_industryco',
                'adbusign','townsign','regtype','TAX_CATEGORIES'
                ]

    # cat_features = ['industryphy',
    #                 'industryco',
    #                 'industryphy_industryco',
    #                 'regtype',
    #                 # 'STATE'
    #             ]
    # del base_info_clean['industryphy']




    # cat_features = [
    #                 'industryco', # 0.8354668580507344
    #                 # 0.8374314212874043
     

    #                 ]

 


    # change_info = pd.read_csv('../data/train/change_info.csv')
    # change_info = change_info.groupby('id',sort=False).agg('count')['bgxmdm']
    # change_info = pd.DataFrame(change_info).reset_index()
    # change_info = change_info.fillna(-1)

    # other_info = pd.read_csv('../data/train/other_info.csv')
    # other_info = other_info.groupby('id',sort=False).agg('mean')
    # other_info = other_info.groupby('id',sort=False).agg('count')
    # other_info = pd.DataFrame(other_info).reset_index()
    # other_info = other_info.fillna(-1)

    # news_info = pd.read_csv('../data/train/news_info.csv')
    # news_score = {}

    # for tup in news_info.itertuples():
    #     if tup[1] not in news_score: 
    #         news_score[tup[1]] = {
    #                             '消极':0,
    #                             '中立':0,
    #                             '积极':0
    #                         }
    #         news_score[tup[1]][tup[2]] += 1
    #     else:
    #         news_score[tup[1]][tup[2]] += 1

    # news_score_id_list = []
    # news_score_neg_list = []
    # news_score_neu_list = []
    # news_score_pos_list = []

    # for k,v in news_score.items():
    #     news_score_id_list.append(k)
    #     news_score_neg_list.append(v['消极'])
    #     news_score_neu_list.append(v['中立'])
    #     news_score_pos_list.append(v['积极'])

    # news_score_dict = {'id':news_score_id_list,
    #                     'news_neg':news_score_neg_list,
    #                     'news_neu':news_score_neu_list,
    #                     'news_pos':news_score_pos_list,
    #                     }
    # news_score_df = pd.DataFrame(news_score_dict, columns=['id','news_neg','news_neu','news_pos'])

    #暂时可以利用企业基本信息，企业纳税信息，企业年度财报信息做义工merge然后进行我们的分类工作
    all_data = base_info_clean.merge(annual_report_info,how='outer')
    all_data = all_data.merge(tax_info_clean,how='outer')


    # all_data = all_data.merge(change_info,how='outer')
    # all_data = all_data.merge(other_info,how='outer')

    # all_data = all_data.merge(news_score_df,how='outer')
    # all_data['news_neg'] = all_data['news_neg'].fillna(0)
    # all_data['news_neu'] = all_data['news_neu'].fillna(0)
    # all_data['news_pos'] = all_data['news_pos'].fillna(0)
    # print(all_data)
  


    all_data = all_data.fillna(-1)




    all_data[cat_features] = all_data[cat_features].astype(int)

    train_df = all_data.merge(entprise_info)
    train_data = train_df.drop(['id','label'],axis=1)
    label = train_df['label']
    test_df = all_data[all_data['id'].isin(entprise_evaluate['id'].unique().tolist())]
    test_df = test_df.reset_index(drop=True)
    return train_data,label, test_df, cat_features


def eval_score(y_test,y_pre):
    _,_,f_class,_=precision_recall_fscore_support(y_true=y_test,y_pred=y_pre,labels=[0,1],average=None)
    fper_class={'合法':f_class[0],'违法':f_class[1],'f1':f1_score(y_test,y_pre)}
    return fper_class

def k_fold_serachParmaters(model,train_val_data,train_val_kind,cate_feature):
    mean_f1 = 0
    mean_f1Train = 0
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    for train, test in sk.split(train_val_data, train_val_kind):
        x_train = train_val_data.iloc[train]
        y_train = train_val_kind.iloc[train]
        x_test = train_val_data.iloc[test]
        y_test = train_val_kind.iloc[test]

        model.fit(x_train, y_train,
                    cat_features=cate_feature,
                    )
        pred = model.predict(x_test)
        fper_class =  eval_score(y_test,pred)
        mean_f1 += fper_class['f1'] / n_splits
        #print(fper_class)
        
        pred_Train = model.predict(x_train)
        fper_class_train =  eval_score(y_train,pred_Train)
        mean_f1Train += fper_class_train['f1']/n_splits
    #print('mean valf1:',mean_f1)
    #print('mean trainf1:',mean_f1Train)
    return mean_f1


def cab_search_param(iter_cnt,lr,max_depth,train_data,label,cat_features):
    clf = cab.CatBoostClassifier(iterations=iter_cnt,
                                learning_rate=lr,
                                depth=max_depth,
                                silent=True,
                                thread_count=12,
                                task_type='CPU',
                                cat_features=cat_features,
                              )
    mean_f1 = k_fold_serachParmaters(clf,train_data,label,cat_features)
    return mean_f1

def rf_search_param(n_estimators,max_depth,min_samples_split):
    rf = RandomForestClassifier(oob_score=True, 
                                random_state=2020,
                                n_estimators= n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                )
    mean_f1 = k_fold_serachParmaters(rf,train_data,label)
    return mean_f1

def lgb_search_param(num_leaves,lr,n_estimators,subsample,colsample_bytree,train_data,label,cat_features):
    lgb = lgb.LGBMClassifier(
        seed=2020,
        num_leaves=64, 
        reg_alpha=0.,
        reg_lambda=0.01, 
        metric='f1',    
        max_depth=-1,
        learning_rate=lr, 
        min_child_samples=10, 

        n_estimators=n_estimators,  # 200,400,800,1000,1500,2000
        subsample=subsample,  # 0.5,0.75,0.9,0.95,1.0
        colsample_bytree=colsample_bytree, # 0.5,0.75,0.9,0.95,1.0
        subsample_freq=1,
        n_jobs=-1,
        min_child_weight=min_child_weight,
        silent=True,


    )

    mean_f1 = k_fold_serachParmaters(lgb,train_data,label)
    return mean_f1

if __name__ == "__main__":
    train_data,label, test_df,cat_features = data_process()
    test_data = test_df.drop(['id'],axis=1)


    # #搜索最佳参数
    # param = []
    # best = 0
    # for iter_cnt in [50,55,60,65,70,75,80,100]:
    #     for lr in [0.005,0.01,0.05,0.075,0.1]:
    #         for max_depth in [7,8,9,10,12,15]:
    #             print('#########',iter_cnt,lr,max_depth)
    #             mean_f1= cab_search_param(iter_cnt,lr,max_depth,train_data,label,cat_features)
    #             if mean_f1 > best:
    #                 param = [iter_cnt,lr,max_depth]
    #                 best = mean_f1
    #                 print(param,best)

    # best param [70, 0.05, 8] 0.8384692833983831
    #[100, 0.05, 8] 0.841834825744929

    # param = []
    # best = 0
    # for n_estimators in [30,50,55,60,65,100]:
    #     print('n_estimators:',n_estimators)
    #     for min_samples_split in [8,10,15,20]:
    #         for max_depth in [6,8,10,13,15]:
    #             mean_f1 = rf_search_param(n_estimators,max_depth,min_samples_split)
    #             if mean_f1 > best:
    #                 param = [n_estimators,min_samples_split,max_depth]
    #                 best = mean_f1
    #                 print(param,best)
    # [55, 8, 10] 0.8283162138324378


    # model = RandomForestClassifier(oob_score=True, 
    #                             random_state=2020,
    #                             n_estimators= 55,
    #                             max_depth =10,
    #                             min_samples_split = 8,
    #                             )

    # model = cab.CatBoostClassifier(iterations=100,
    #                             learning_rate=0.05,
    #                             depth=8,
    #                             silent=True,
    #                             thread_count=8,
    #                             task_type='CPU',
    #                             cat_features=cat_features,
    #                           )

 
    model = cab.CatBoostClassifier(iterations=70,
                                learning_rate=0.05,
                                depth=8,
                                silent=True,
                                thread_count=8,
                                task_type='CPU',
                                cat_features=cat_features,
                              )  

    details = []
    answers = []
    mean_f1 = 0
    n_splits = 5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    cnt = 0
    for train, test in sk.split(train_data, label):
        x_train = train_data.iloc[train]
        y_train = label.iloc[train]
        x_test = train_data.iloc[test]
        y_test = label.iloc[test]

        model.fit(x_train, y_train)
        pred_cab = model.predict(x_test)
        weight_cab =  eval_score(y_test,pred_cab)['f1']

        print('每{}次验证的f1:{}'.format(cnt,weight_cab))
        cnt += 1
        mean_f1+=weight_cab/n_splits
        ans = model.predict_proba(test_data)

        answers.append(ans)
    print('mean f1:',mean_f1)

    #fina=sum(answers)/n_splits#
    fina = np.sqrt(sum(np.array(answers)**2)/n_splits)#平方平均
    fina = fina[:,1]
    test_df['score'] = fina#可选:fina_persudo是伪标签的预测结果
    submit_csv = test_df[['id','score']]
    submit_csv.to_csv('../result/11_10/submit_'+str(version)+'.csv', index=False) #submit

