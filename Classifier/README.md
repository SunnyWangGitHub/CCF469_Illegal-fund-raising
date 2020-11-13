# CCF469_Illegal-fund-raising
CCF大数据与计算智能大赛-企业非法集资风险预测

### Data
### 数据分析
    具有年报基本信息的企业中，有536违法；2800合法；5601为测试集
    具有年报基本信息的企业中：合法/违法:5.223880597014926
    不具有年报基本信息的企业中：合法/违法:24.907865168539328


    具有纳税基本信息的企业中，有75违法；99合法；634为测试集
    具有纳税信息的企业中：合法/违法:1.32
    不具纳税信息的企业中：合法/违法:15.21523178807947


### 提交示例
    id, score
    XXXXXX,0.1246
    XXXXXX,0.8796


### Reference
    > 知乎baseline https://zhuanlan.zhihu.com/p/267119113
    > https://github.com/DLLXW/data-science-competition/blob/main/datafountain/baseline_RandomForest829.ipynb

### Log
    train_v0   0.82455582
    base + annual_report_info(2018、未做处理)

    train_v1    0.83154018669  
    base + annual_report_info(2018、做了处理)  fillna 0.82114447461
    ==> bug! 
    annual_report_info['BUSSTNAME'] = annual_report_info['BUSSTNAME'].map({'开业': 0,'歇业': 1,'停业': 1}).map({'开业': 0,'歇业': 1,'停业': 1})

                # annual_report_info = annual_report_info.fillna(-1)   0.81297185830 
 
                # 只取2018  NAN不做处理 0.83157109675  微涨

                            NAN做了处理         0.81916130109
    
    train_v0 
    rf:
    offline : 0.8283162138324378
    online :  0.82007679627 

    cab:
    offline : 0.8384692833983831  0.8384692833983831
    online: 0.83199645665  0.83199645665
    
    offline :0.841834825744929     /0.8400464197671225   / 0.8338829484468404
    online:0.82810406341          /0.82572223934        /0.82554387552 

    train_v0 : 0.83199645665
