# CCF469_Illegal-fund-raising
CCF大数据与计算智能大赛-企业非法集资风险预测

### Data
#### 数据简介
    该数据集包含约25000家企业数据，其中约15000家企业带标注数据作为训练集，剩余数据作为测试集。数据由企业基本信息、企业年报、企业纳税情况等组成，数据包括数值型、字符型、日期型等众多数据类型（已脱敏），部分字段内容在部分企业中有缺失，其中第一列id为企业唯一标识。

### 数据说明
    数据集1：base_info.csv
    包含数据集7和8中涉及到的所有企业的基本信息，每一行代表一个企业的基本数据，每一行有33列，其中id列为企业唯一标识，列之间采用“,”分隔符分割。
    数据格式如下：
    [id:企业唯一标识, oplocdistrict:行政区划代码, industryphy:行业类别代码, industryco:行业细类代码, dom:经营地址, opscope:经营范围, enttype:企业类型, enttypeitem:企业类型小类, opfrom:经营期限起, opto:经营期限止, state:状态, orgid:机构标识, jobid:职位标识, adbusign:是否广告经营, townsign:是否城镇, regtype:主题登记类型, empnum:从业人数, compform:组织形式, parnum:合伙人数, exenum:执行人数, opform:经营方式, ptbusscope:兼营范围, venind:风险行业, enttypeminu:企业类型细类, midpreindcode:中西部优势产业代码, protype:项目类型, oploc:经营场所, regcap:注册资本（金）, reccap:实缴资本, forreccap:实缴资本（外方）, forregcap:注册资本（外方）, congro:投资总额, enttypegb:企业（机构）类型]


    数据集2：annual_report_info.csv
    包含数据集7和8中涉及到的企业的年报基本信息，每一行代表一个企业的年报基本数据，每一行有23列，其中id列为企业唯一标识，列之间采用“,”分隔符分割。
    数据格式如下：
    [id:企业唯一标识, ANCHEYEAR:年度, STATE:状态, FUNDAM:资金数额, MEMNUM:成员人数, FARNUM:农民人数, ANNNEWMEMNUM:本年度新增成员人数, ANNREDMEMNUM:本年度退出成员人数, EMPNUM:从业人数, EMPNUMSIGN:从业人数是否公示, BUSSTNAME:经营状态名称, COLGRANUM:其中高校毕业生人数经营者, RETSOLNUM:其中退役士兵人数经营者, DISPERNUM:其中残疾人人数经营者, UNENUM:其中下岗失业人数经营者, COLEMPLNUM:其中高校毕业生人数雇员, RETEMPLNUM:其中退役士兵人数雇员, DISEMPLNUM:其中残疾人人数雇员, UNEEMPLNUM:其中下岗失业人数雇员, WEBSITSIGN:是否有网站标志, FORINVESTSIGN:是否有对外投资企业标志, STOCKTRANSIGN:有限责任公司本年度是否发生股东股权转让标志, PUBSTATE:公示状态：1 全部公示，2部分公示,3全部不公示]


    数据集3：tax_info.csv
    包含数据集7和8中涉及到的企业的纳税信息，每一行代表一个企业的纳税信息，每一行有9列，其中id列为企业唯一标识，列之间采用“,”分隔符分割。
    数据格式如下：
    [id:企业唯一标识, START_DATE:起始时间, END_DATE:终止时间, TAX_CATEGORIES:税种, TAX_ITEMS:税目, TAXATION_BASIS:计税依据, TAX_RATE:税率, DEDUCTION:扣除数, TAX_AMOUNT:税额]


    数据集4：change_info.csv
    包含数据集7和8中涉及到的企业的变更信息，每一行代表一个企业变更信息，每一行5列，其中id列为企业唯一标识，列之间采用“,”分隔符分割。
    数据格式如下：
    [id:企业唯一标识, bgxmdm:变更信息代码, bgq:变更前, bgh:变更后, bgrq:变更日期]


    数据集5：news_info.csv
    包含数据集7和8中涉及到的企业的新闻舆情信息，每一行代表一个企业新闻舆情，每一行3列，其中id列为企业唯一标识，列之间采用“,”分隔符分割。
    数据格式如下：
    [id:企业唯一标识, positive_negtive:新闻正负面性, public_date:发布日期]


    数据集6：other_info.csv
    包含数据集7和8中涉及到的企业的其他信息，每一行代表一个企业其他信息，每一行4列，其中id列为企业唯一标识，列之间采用“,”分隔符分割。
    数据格式如下：
    [id:企业唯一标识, legal_judgment_num:裁判文书数量, brand_num:注册商标数量, patent_num:专利数量]


    数据集7：entprise_info.csv
    带标注的企业数据。每一行代表一个企业，每一行2列，其中id列为企业唯一标识，label列为标注（1：有非法集资风险，0：无非法集资风险），列之间采用“,”分隔符分割。


    数据集8（验证集）：entprise_evaluate.csv
    未标注企业数据。参赛队伍需提交的最终结果数据集，每一行代表一个企业，每一行有 2 列, 其中id列为企业唯一标识，score列为空，列之间采用“,”分隔符分割。

### 提交要求
    参赛者以csv文件提交：参赛者以csv文件格式，提交模型结果到大数据竞赛平台，平台进行在线评分，实时排名。目前平台仅支持单文件提交，即所有提交内容需要放在一个文件中。

### 提交格式
    参赛队伍需依据提供的数据集8，给出企业是否有非法集资风险的预测概率值，每条预测需提供 2 列，列之间采用 “,” 分隔符分割。
    请注意：请勿改变验证集列的顺序，请按照给出的文件的列顺序加入预测值。
    [id, score]
    1.企业id
    2.预测的非法集资风险概率值，取值范围 [0, 1]

### 提交示例
    id, score
    XXXXXX,0.1246
    XXXXXX,0.8796


### Reference
    > 知乎baseline https://zhuanlan.zhihu.com/p/267119113