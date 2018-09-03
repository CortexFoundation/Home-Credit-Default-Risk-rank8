from tool.tool import *


cache_path = 'E:/credit/'
data_path = 'C:/Users/cui/Desktop/python/credit/data/'
inplace = False


############################### 工具函数 ###########################
# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result
# 统计转化率
def convert(data,stat,key):
    key = key if type(key) == list else [key]
    stat_temp = stat[key + ['label']].copy()
    rate = stat_temp.groupby(key,as_index=False)['label'].agg({'sum':'sum', 'count':'count'})
    rate['_'.join(key)+'_convert'] = ((rate['sum']+4)/(rate['count']+46))
    data = data.merge(rate, on=key, how='left')
    return data['_'.join(key)+'_convert']
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True, min_count=100,inplace=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    result = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in result.columns if c not in original_columns]
    cat_columns = [c for c in original_columns if c not in result.columns]
    if not inplace:
        for c in cat_columns:
            result[c] = df[c]
    for c in new_columns:
        if (result[c].sum()<100) or ((result.shape[0]-result[c].sum())<100):
            del result[c]
            new_columns.remove(c)
    return result, new_columns


############################### 特征函数 ###########################
def get_attribute(data,data_key):
    result_path = cache_path + 'attribute_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        data['basementarea_apartments_ratio'] = data['BASEMENTAREA_AVG'] / (
        data['APARTMENTS_AVG'] + 0.01)  # 地下室面积占地上面积比例
        data['second_hours_years'] = data['YEARS_BEGINEXPLUATATION_AVG'] / (
        data['YEARS_BUILD_AVG'] + 0.01)  # 房子是否为二手房
        data['AMT_INCOME_TOTAL/CNT_CHILDREN_ratio'] = np.log1p(data['AMT_INCOME_TOTAL']) / (
        data['CNT_CHILDREN'] + 1)  # 养孩子负担
        data['NEW_INC_PER_CHLD'] = data['AMT_INCOME_TOTAL'] / (1 + data['CNT_CHILDREN'])  # 养孩子负担2
        data['AMT_INCOME_TOTAL_&_CNT_FAM_MEMBERS_ratio'] = data['AMT_INCOME_TOTAL'] / (
        1 + data['CNT_FAM_MEMBERS'])  # 养家负担
        data['other_family_count'] = data['CNT_FAM_MEMBERS'] - data['CNT_CHILDREN']  # 其他成年人
        data['AMT_CREDIT_&_other_family_count_ratio'] = data['AMT_CREDIT'] / (data['other_family_count'])  # 家人还款压力
        data['AMT_CREDIT_&_CNT_FAM_MEMBERS_ratio'] = data['AMT_CREDIT'] / (data['CNT_FAM_MEMBERS'])  # 养家压力
        data['CNT_CHILDREN_&_other_family_count_ratio'] = data['CNT_CHILDREN'] / (
        data['other_family_count'])  # 小孩数量比成年人数量

        data['AMT_CREDIT_&_AMT_ANNUITY_ratio'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']  # 单月贷款额度/每年贷款额度
        data['AMT_CREDIT_&_AMT_ANNUITY_diff'] = data['AMT_CREDIT'] - data['AMT_ANNUITY']  # 单月贷款额度-每年贷款额度
        data['AMT_CREDIT_&_AMT_GOODS_PRICE_ratio'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']  # 单月贷款额度/物品价格
        data['AMT_CREDIT_&_AMT_INCOME_TOTAL_ratio'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']  # 单月贷款额度/每年收入
        data['AMT_INCOME_TOTAL_&_AMT_ANNUITY_ratio'] = data['AMT_INCOME_TOTAL'] / data[
            'AMT_GOODS_PRICE']  # 物品价格/每年收入
        # 房屋价格

        data['DAYS_BIRTH_&_DAYS_EMPLOYED_diff'] = data['DAYS_BIRTH'] - data['DAYS_EMPLOYED']
        data['DAYS_BIRTH_&_DAYS_REGISTRATION_diff'] = data['DAYS_BIRTH'] - data['DAYS_REGISTRATION']
        data['DAYS_BIRTH_&_DAYS_ID_PUBLISH_diff'] = data['DAYS_BIRTH'] - data['DAYS_ID_PUBLISH']
        data['DAYS_BIRTH_&_DAYS_LAST_PHONE_CHANGE_diff'] = data['DAYS_BIRTH'] - data['DAYS_LAST_PHONE_CHANGE']
        data['DAYS_BIRTH_&_OWN_CAR_AGE_diff'] = data['DAYS_BIRTH'] - data['OWN_CAR_AGE'] * 365

        FLAG_DOCUMENT_ = [c for c in data.columns if 'FLAG_DOCUMENT_' in c]
        data['FLAG_DOCUMENT_sum'] = data[FLAG_DOCUMENT_].sum(axis=1)

        data['FLAG_PHONE'] = 1 - data['FLAG_PHONE']
        data['FLAG_EMAIL'] = 1 - data['FLAG_EMAIL']
        FLAG_ = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
        data['FLAG_sum'] = data[FLAG_].sum(axis=1)

        EXT_SOURCE_ = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        data['EXT_SOURCE_mean'] = data[EXT_SOURCE_].mean(axis=1)
        data['EXT_SOURCE_min'] = data[EXT_SOURCE_].min(axis=1)
        data['EXT_SOURCE_max'] = data[EXT_SOURCE_].max(axis=1)
        data['EXT_SOURCE_std'] = data[EXT_SOURCE_].std(axis=1)
        data['NEW_SOURCES_PROD'] = data['EXT_SOURCE_1'] * data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
        data['NEW_SOURCES_PROD-1'] = (1 - data['EXT_SOURCE_1']) * (1 - data['EXT_SOURCE_2']) * (
        1 - data['EXT_SOURCE_3'])

        inc_by_org = data[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()[
            'AMT_INCOME_TOTAL']
        data['NEW_INC_BY_ORG'] = data['ORGANIZATION_TYPE'].map(inc_by_org)
        data['NEW_EMPLOY_TO_BIRTH_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
        data['NEW_ANNUITY_TO_INCOME_RATIO'] = data['AMT_ANNUITY'] / (1 + data['AMT_INCOME_TOTAL'])

        data['NEW_CAR_TO_BIRTH_RATIO'] = data['OWN_CAR_AGE'] / data['DAYS_BIRTH']
        data['NEW_CAR_TO_EMPLOY_RATIO'] = data['OWN_CAR_AGE'] / data['DAYS_EMPLOYED']
        data['NEW_PHONE_TO_BIRTH_RATIO'] = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_BIRTH']
        data['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_EMPLOYED']
        data['NEW_CREDIT_TO_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
        categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
        data = pd.get_dummies(data, columns=categorical_columns, dummy_na=1000)
        for c in data.columns:
            if data[c].dtype == 'object':
                # data[c] = convert(data, data, c)
                data[c + '_count'] = data[c].map(data.groupby(c).size())
        data.reset_index(drop=True,inplace=True)
        data.columns = ['app_' + c if c not in ['SK_ID_CURR', 'label'] else c for c in data.columns]
        data.to_feather(result_path)
    return data

def get_attribute2(data,data_key):
    result_path = cache_path + 'attribute2_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        df = data
        docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
        live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

        inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()[
            'AMT_INCOME_TOTAL']

        df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
        df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
        df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
        df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
        df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
        df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
        df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
        df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
        df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
        df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

        # Categorical features with Binary encode (0 or 1; two categories)
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])
        # Categorical features with One-Hot encode
        df, cat_cols = one_hot_encoder(df, False)
        dropcolum = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4',
                     'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                     'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                     'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                     'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                     'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                     'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
        data = df.drop(dropcolum, axis=1)
        data.reset_index(drop=True,inplace=True)
        data.to_feather(result_path)
    return data


# bureau特征
def get_bureau_and_balance_feat(ids,data_key):
    result_path = cache_path + 'bureau_and_balance_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        data = ids.copy()
        bureau = pd.read_csv(data_path + 'bureau.csv').sort_values('DAYS_CREDIT',ascending=True)
        bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=True, min_count=100, inplace=False)
        bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] < bureau['DAYS_CREDIT'], 'DAYS_CREDIT_ENDDATE'] = np.nan
        data['id_count'] = data['SK_ID_CURR'].map(bureau['SK_ID_CURR'].value_counts()).fillna(0)
        bureau['DAYS_ENDDATE_FACT_diff'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
        bureau['DAYS_CREDIT-DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
        bureau['money_perday'] =  bureau['AMT_CREDIT_SUM'] / bureau['DAYS_CREDIT-DAYS_CREDIT_ENDDATE']
        bureau['AMT_CREDIT_SUM_diff'] = bureau['AMT_CREDIT_SUM'] - bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].shift( 1)
        bureau['AMT_CREDIT_SUM_DEBT_diff'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
        bureau['DAYS_CREDIT_UPDATE_diff'] = bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_CREDIT']

        # for i in [0,1,2,3,4,5,6]:
        #     data['CREDIT_TYPE_last_time{}'.format(i)] = get_last_values(data,bureau,'SK_ID_CURR','DAYS_CREDIT','DAYS_CREDIT',i)
        # for i in [0, 1, 2, 3,4,5,6]:
        #     data['CREDIT_TYPE_first_time{}'.format(i)] = get_first_values(data,bureau,'SK_ID_CURR','DAYS_CREDIT','DAYS_CREDIT',i)

        agg = {
            'DAYS_CREDIT':['sum', 'mean','std','median','skew'],
            'DAYS_CREDIT_ENDDATE': ['min','max','mean','sum'],
            'DAYS_ENDDATE_FACT_diff':['min','max','mean','sum'],
            'DAYS_CREDIT-DAYS_CREDIT_ENDDATE':['min','max','mean','sum'],
            'money_perday':['min','max','mean'],
            'AMT_CREDIT_MAX_OVERDUE':['max','min','sum','mean'],
            'CNT_CREDIT_PROLONG':['max','sum'],
            'AMT_CREDIT_SUM':['min','max','mean','sum','std','median','skew','last','first',np.ptp],
            'AMT_CREDIT_SUM_diff':['min','max','mean','sum','skew','std','last','first'],
            'CREDIT_DAY_OVERDUE':['max','sum',],
            'AMT_CREDIT_SUM_DEBT':['min','max','mean','sum','std','median','skew','last','first',np.ptp],
            'AMT_CREDIT_SUM_DEBT_diff':['min','max','mean','sum','std','median','skew','last','first',ss.kurtosis,np.ptp],
            'AMT_CREDIT_SUM_OVERDUE':['count','max','sum','mean'],
            'AMT_CREDIT_SUM_LIMIT':['min','max','mean','sum'],
            'DAYS_CREDIT_UPDATE':['min','max','mean','std','median','skew','last','first'],
            'DAYS_CREDIT_UPDATE_diff':['min','max','mean','std','median','skew','last','first'],
            'AMT_ANNUITY':['min','max','mean']
        }
        for c in bureau_cat: agg[c]=['mean','sum']

        bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg)
        bureau_agg.columns = pd.Index([e[0] + "_" + e[1] for e in bureau_agg.columns.tolist()])
        data = data.merge(bureau_agg.reset_index(), on='SK_ID_CURR', how='left')

        for days in [1462,730,365,183]:
            stat = bureau[bureau['DAYS_CREDIT']>-days].copy()
            agg = {
                'DAYS_CREDIT': ['count','mean','skew'],
                'DAYS_ENDDATE_FACT_diff':['max'],
                'AMT_CREDIT_MAX_OVERDUE': ['max'],
                'AMT_CREDIT_SUM_DEBT_diff': ['min', 'max', 'mean', 'sum', 'std', 'median', 'skew', 'last', 'first', ss.kurtosis, np.ptp],
                'AMT_CREDIT_SUM':['sum'],
                'AMT_CREDIT_SUM_DEBT':['sum'],
                'AMT_CREDIT_SUM_LIMIT':['mean'],
            }

            stat_agg = stat.groupby('SK_ID_CURR').agg(agg)
            stat_agg.columns = pd.Index([e[0] + "_" + e[1] for e in stat_agg.columns.tolist()])
            stat_agg['AMT_CREDIT_SUM_DEBT_sum_ratio'] = stat_agg['AMT_CREDIT_SUM_sum'] / (stat_agg['AMT_CREDIT_SUM_DEBT_sum'] + 0.01)
            stat_agg.columns = [c + '_{}days_ago'.format(days) for c in stat_agg.columns]
            data = data.merge(stat_agg.reset_index(), on='SK_ID_CURR', how='left')

        stat = bureau[['SK_ID_CURR','DAYS_CREDIT','DAYS_CREDIT_ENDDATE','money_perday']].copy()
        stat['DAYS_CREDIT'] = stat['DAYS_CREDIT']//30
        stat['DAYS_CREDIT_ENDDATE'] = stat['DAYS_CREDIT_ENDDATE'] // 30
        for i in range(97):
            stat[i] = ((stat['DAYS_CREDIT']<i) & (stat['DAYS_CREDIT_ENDDATE']>=i)).astype(int) * stat['money_perday']
        stat.drop(['DAYS_CREDIT','DAYS_CREDIT_ENDDATE','money_perday'],axis=1,inplace=True)
        stat2 = (stat.set_index('SK_ID_CURR')>0).reset_index()
        data['own_month_count_sum'] = data['SK_ID_CURR'].map(stat2.groupby('SK_ID_CURR').sum().sum(axis=1))
        data['own_month_count_max'] = data['SK_ID_CURR'].map(stat2.groupby('SK_ID_CURR').sum().max(axis=1))
        data['own_money_count_sum'] = data['SK_ID_CURR'].map(stat.groupby('SK_ID_CURR').sum().max(axis=1))
        stat = stat.groupby('SK_ID_CURR').sum()
        for i in [1,3,6,12,24,48,96]:
            data['own_money_sum_{}month'.format(i)] = data['SK_ID_CURR'].map(stat[list(range(i))].sum(axis=1))
            if i>2:
                data['own_money_trend_{}month'.format(i)] = data['SK_ID_CURR'].map(stat[list(range(i))].apply(trend,axis=1))


        obj_columns = [c for c in bureau.columns if bureau[c].dtype == 'object']
        bureau.groupby('SK_ID_CURR')[obj_columns]

        data['AMT_CREDIT_MAX_OVERDUE_count'] = groupby(bureau, bureau[bureau['AMT_CREDIT_MAX_OVERDUE'] > 0],
                                                       'SK_ID_CURR', 'AMT_CREDIT_MAX_OVERDUE', 'count')
        data['CNT_CREDIT_PROLONG_count'] = groupby(bureau, bureau[bureau['CNT_CREDIT_PROLONG'] > 0], 'SK_ID_CURR',
                                                   'CNT_CREDIT_PROLONG', 'count')
        data['AMT_CREDIT_SUM_last-first'] = data['AMT_CREDIT_SUM_last'] - data['AMT_CREDIT_SUM_first']


        data['AMT_CREDIT_SUM_diff_last'] = get_last_values(data, bureau[~bureau['AMT_CREDIT_SUM_diff'].isnull()],
                                                           'SK_ID_CURR', 'DAYS_CREDIT', 'AMT_CREDIT_SUM_diff', 0)
        data['AMT_CREDIT_SUM_diff_first'] = get_first_values(data, bureau[~bureau['AMT_CREDIT_SUM_diff'].isnull()],
                                                             'SK_ID_CURR', 'DAYS_CREDIT', 'AMT_CREDIT_SUM_diff', 0)
        data['AMT_CREDIT_SUM_diff_last-first'] = data['AMT_CREDIT_SUM_diff_last'] - data['AMT_CREDIT_SUM_diff_first']

        data['AMT_CREDIT_SUM_DEBT_count'] = groupby(data, bureau[bureau['AMT_CREDIT_SUM_DEBT'] > 0], 'SK_ID_CURR',
                                                    'AMT_CREDIT_SUM_DEBT', 'count')
        data['AMT_CREDIT_SUM_DEBT_sum_ratio'] = data['AMT_CREDIT_SUM_sum'] / (data['AMT_CREDIT_SUM_DEBT_sum'] + 0.01)

        data['AMT_CREDIT_SUM_DEBT_diff_count'] = groupby(data, bureau[~bureau['AMT_CREDIT_SUM_DEBT_diff'].isnull()],
                                                         'SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT_diff', 'count')

        data['AMT_CREDIT_SUM_OVERDUE_sum_ratio1'] = data['AMT_CREDIT_SUM_OVERDUE_sum'] / (
                    data['AMT_CREDIT_SUM_DEBT_sum'] + 0.01)
        data['AMT_CREDIT_SUM_OVERDUE_sum_ratio2'] = data['AMT_CREDIT_SUM_OVERDUE_sum'] / (
                    data['AMT_CREDIT_SUM_sum'] + 0.01)

        #'Closed', 'Active', 'Sold', 'Bad debt'
        stat = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].copy()
        data['Closed_count'] = groupby(data,stat,'SK_ID_CURR','DAYS_CREDIT','count').fillna(0)
        data['Closed_AMT_CREDIT_SUM_sum'] = groupby(data, stat, 'SK_ID_CURR', 'AMT_CREDIT_SUM', 'sum').fillna(0)
        data['Closed_count_end1'] = groupby(data, stat[stat['DAYS_CREDIT_ENDDATE']>0], 'SK_ID_CURR', 'DAYS_CREDIT', 'count').fillna(0)
        data['Closed_count_end2'] = groupby(data, stat[stat['DAYS_CREDIT_ENDDATE'] < 0], 'SK_ID_CURR', 'DAYS_CREDIT', 'count').fillna(0)
        # data['Closed_CREDIT_TYPE_last_time{}'.format(0)] = get_last_values(data, stat, 'SK_ID_CURR','DAYS_CREDIT','DAYS_CREDIT',0)
        # data['Closed_CREDIT_TYPE_first_time{}'.format(0)] = get_first_values(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT','DAYS_CREDIT', 0)

        stat = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].copy()
        data['Active_count'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT', 'count').fillna(0)
        data['Active_AMT_CREDIT_SUM_sum'] = groupby(data, stat, 'SK_ID_CURR', 'AMT_CREDIT_SUM', 'sum').fillna(0)
        data['Active_count_end1'] = groupby(data, stat[stat['DAYS_CREDIT_ENDDATE'] > 0], 'SK_ID_CURR', 'DAYS_CREDIT', 'count').fillna(0)
        data['Active_count_end2'] = groupby(data, stat[stat['DAYS_CREDIT_ENDDATE'] < 0], 'SK_ID_CURR', 'DAYS_CREDIT', 'count').fillna(0)
        data['Active_DAYS_CREDIT_sum'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT', 'sum')
        data['Active_DAYS_CREDIT_ENDDATE_sum'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT_ENDDATE', 'sum')
        data['Active_DAYS_CREDIT-DAYS_CREDIT_ENDDATE_sum'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT-DAYS_CREDIT_ENDDATE', 'sum')
        data['Active_DAYS_CREDIT_sum_ratio'] = data['Active_DAYS_CREDIT_ENDDATE_sum'] / (data['Active_DAYS_CREDIT_sum']+0.1)


        # data['Active_CREDIT_TYPE_last_time{}'.format(0)] = get_last_values(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT', 'DAYS_CREDIT',0)
        # data['Active_CREDIT_TYPE_first_time{}'.format(0)] = get_first_values(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT','DAYS_CREDIT', 0)
        data['Closed_count_&_Active_count_ratio'] = data['Closed_count'] / (data['Active_count']+0.01)
        data['Closed_AMT_CREDIT_SUM_sum_&_Active_AMT_CREDIT_SUM_sum_ratio'] = data['Closed_AMT_CREDIT_SUM_sum'] / (data['Active_AMT_CREDIT_SUM_sum']+0.01)


        stat = bureau[bureau['CREDIT_ACTIVE'] == 'Sold'].copy()
        data['Sold_count'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT', 'count').fillna(0)
        # data['Sold_CREDIT_TYPE_last_time{}'.format(0)] = get_last_values(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT', 'DAYS_CREDIT', 0)
        # data['Sold_CREDIT_TYPE_first_time{}'.format(0)] = get_first_values(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT','DAYS_CREDIT', 0)

        # stat = bureau[bureau['CREDIT_ACTIVE'] == 'Bad debt'].copy()
        # data['Bad_debt_CREDIT_TYPE_last_time{}'.format(0)] = get_last_values(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT', 'DAYS_CREDIT', 0)


        stat = bureau[bureau['DAYS_CREDIT_ENDDATE'] < 0].copy()
        data['DAYS_CREDIT-DAYS_CREDIT_ENDDATE_closed_mean'] = groupby(data,stat, 'SK_ID_CURR','DAYS_CREDIT-DAYS_CREDIT_ENDDATE','mean')
        data['DAYS_CREDIT-DAYS_CREDIT_ENDDATE_closed_max'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT-DAYS_CREDIT_ENDDATE', 'max')
        data['DAYS_CREDIT-DAYS_CREDIT_ENDDATE_closed_min'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT-DAYS_CREDIT_ENDDATE', 'min')
        stat = bureau[bureau['DAYS_CREDIT_ENDDATE'] > 0].copy()
        data['DAYS_CREDIT-DAYS_CREDIT_ENDDATE_active_mean'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT-DAYS_CREDIT_ENDDATE', 'mean')
        data['DAYS_CREDIT-DAYS_CREDIT_ENDDATE_active_max'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT-DAYS_CREDIT_ENDDATE', 'max')
        data['DAYS_CREDIT-DAYS_CREDIT_ENDDATE_active_min'] = groupby(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT-DAYS_CREDIT_ENDDATE', 'min')


        data['n_CREDIT_TYPE'] = groupby(data, bureau, 'SK_ID_CURR', 'CREDIT_TYPE', 'nunique')
        stat = bureau[bureau['CREDIT_TYPE'].isin(['Consumer credit', 'Credit card', 'Car loan', 'Mortgage', 'Microloan',
       'Loan for business development', 'Another type of loan','Unknown type of loan', 'Loan for working capital replenishment',
       'Cash loan (non-earmarked)', 'Real estate loan', 'Loan for the purchase of equipment',])].copy()
        stat2 = stat.groupby(['SK_ID_CURR','CREDIT_TYPE']).size().unstack().add_suffix('_count').reset_index()
        data = data.merge(stat2,on='SK_ID_CURR',how='left')
        stat2 = stat.groupby(['SK_ID_CURR', 'CREDIT_TYPE'])['DAYS_CREDIT'].max().unstack().add_suffix('_last_day').reset_index()
        data = data.merge(stat2, on='SK_ID_CURR', how='left')
        stat2 = stat.groupby(['SK_ID_CURR', 'CREDIT_TYPE'])['DAYS_CREDIT'].min().unstack().add_suffix('_first_day').reset_index()
        data = data.merge(stat2, on='SK_ID_CURR', how='left')
        stat2 = stat.groupby(['SK_ID_CURR', 'CREDIT_TYPE'])['AMT_CREDIT_SUM'].sum().unstack().add_suffix( '_sum').reset_index()
        data = data.merge(stat2, on='SK_ID_CURR', how='left')
        stat2 = stat.groupby(['SK_ID_CURR', 'CREDIT_TYPE'])['AMT_CREDIT_SUM'].max().unstack().add_suffix('_max').reset_index()
        data = data.merge(stat2, on='SK_ID_CURR', how='left')
        stat2 = stat.groupby(['SK_ID_CURR', 'CREDIT_TYPE'])['AMT_CREDIT_SUM'].mean().unstack().add_suffix('_mean').reset_index()
        data = data.merge(stat2, on='SK_ID_CURR', how='left')


        stat = bureau[bureau['AMT_ANNUITY'] > 0].copy()
        data['AMT_ANNUITY_count'] = groupby(data, stat, 'SK_ID_CURR', 'AMT_ANNUITY', 'count')
        data['AMT_ANNUITY_min'] = groupby(data, stat, 'SK_ID_CURR', 'AMT_ANNUITY', 'min')
        data['AMT_ANNUITY_max'] = groupby(data, stat, 'SK_ID_CURR', 'AMT_ANNUITY', 'max')
        data['AMT_ANNUITY_mean'] = groupby(data, stat, 'SK_ID_CURR', 'AMT_ANNUITY', 'mean')
        data['AMT_ANNUITY_std'] = groupby(data, stat, 'SK_ID_CURR', 'AMT_ANNUITY', 'std')
        data['AMT_ANNUITY_last'] = get_last_values(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT', 'AMT_ANNUITY', 0)
        data['AMT_ANNUITY_first'] = get_first_values(data, stat, 'SK_ID_CURR', 'DAYS_CREDIT','AMT_ANNUITY', 0)

        stat = bureau[['SK_ID_CURR','SK_ID_BUREAU']].copy()
        bureau_balance = pd.read_csv(data_path + 'bureau_balance.csv')
        bureau_balance,new_columns = one_hot_encoder(bureau_balance, nan_as_category=True, min_count=100, inplace=False)
        agg = bureau_balance.groupby('SK_ID_BUREAU')[new_columns].mean()
        stat = stat.merge(agg.reset_index(),on='SK_ID_BUREAU',how='left')
        agg = {c:['min','max','mean'] for c in new_columns}
        stat2 = bureau_balance[bureau_balance['STATUS']=='C'].copy()
        stat['C_count'] = groupby(stat, stat2,'SK_ID_BUREAU', 'SK_ID_BUREAU','count')
        stat['C_min'] = groupby(stat, stat2, 'SK_ID_BUREAU', 'MONTHS_BALANCE', 'min')
        stat['C_max'] = groupby(stat, stat2, 'SK_ID_BUREAU', 'MONTHS_BALANCE', 'max')
        stat2 = bureau_balance[bureau_balance['STATUS'] == 'X'].copy()
        stat['X_count'] = groupby(stat, stat2, 'SK_ID_BUREAU', 'SK_ID_BUREAU', 'count')
        stat['X_min'] = groupby(stat, stat2, 'SK_ID_BUREAU', 'MONTHS_BALANCE', 'min')
        stat['X_max'] = groupby(stat, stat2, 'SK_ID_BUREAU', 'MONTHS_BALANCE', 'max')
        stat2 = bureau_balance[bureau_balance['STATUS'].isin(['0', '1', '2', '3', '5', '4'])].sort_values('STATUS').drop_duplicates('SK_ID_BUREAU', keep='last')
        stat2['STATUS'] = stat2['STATUS'].astype(int)
        stat2 = stat2[['SK_ID_BUREAU','MONTHS_BALANCE','STATUS']]
        stat = stat.merge(stat2, on='SK_ID_BUREAU',how='left').rename(columns={'MONTHS_BALANCE':'max_STATUS_last_time','STATUS':'max_STATUS'})

        agg.update({
            'C_count': ['sum', 'count'],
            'C_min': ['min', 'max'],
            'C_max': ['min', 'max'],
            'X_count': ['sum', 'count'],
            'X_min': ['min', 'max'],
            'X_max': ['min', 'max'],
            'max_STATUS': ['max'],
            'max_STATUS_last_time': ['max'],
        })
        agg = stat.groupby('SK_ID_CURR').agg(agg)
        agg.columns = [e[0] + "_" + e[1] for e in agg.columns.tolist()]
        data = data.merge(agg.reset_index(), on='SK_ID_CURR', how='left')
        data.columns = ['bureau_' + c if c not in ['SK_ID_CURR', 'label'] else c for c in data.columns]
        data.to_feather(result_path)
    return data


# bureau特征
def get_bureau_and_balance_feat2(ids,data_key):
    result_path = cache_path + 'bureau_and_balance2_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 0:
        data = pd.read_feather(result_path)
    else:
        bureau = pd.read_csv(data_path + 'bureau.csv')
        bb = pd.read_csv(data_path + 'bureau_balance.csv')
        bb, bb_cat = one_hot_encoder(bb, False)
        bureau, bureau_cat = one_hot_encoder(bureau, False)

        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1] for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        del bb, bb_agg

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_min': ['min'],
            'MONTHS_BALANCE_max': ['max'],
            'MONTHS_BALANCE_size': ['mean', 'sum']
        }
        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat: cat_aggregations[cat] = ['mean']
        for cat in bb_cat: cat_aggregations[cat + "_mean"] = ['mean']

        bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index([e[0] + "_" + e[1] for e in bureau_agg.columns.tolist()])
        # # Bureau: Active credits - using only numerical aggregations
        # active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        # active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        # active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        # bureau_agg = bureau_agg.join(active_agg, how='left')
        # del active, active_agg
        # Bureau: Closed credits - using only numerical aggregations
        # closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        # closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        # closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        # bureau_agg = bureau_agg.join(closed_agg, how='left')
        data = ids.merge(bureau_agg.reset_index(),on='SK_ID_CURR',how='left')
        data.to_feather(result_path)
    return data

# POS_CASH_balance特征
def get_POS_CASH_balance_feat(ids,data_key):
    result_path = cache_path + 'POS_CASH_balance_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        data = ids.copy()
        pcb = pd.read_csv(data_path + 'POS_CASH_balance.csv').sort_values('MONTHS_BALANCE')
        pcb['SK_DPD_rate'] = (pcb['SK_DPD'] > 0).astype(int)
        pcb['SK_DPD_DEF_rate'] = (pcb['SK_DPD_DEF'] > 0).astype(int)
        pcb['SK_DPD_diff'] = pcb['SK_DPD'] - pcb['SK_DPD_DEF']
        agg = {
            'SK_ID_PREV': ['nunique', 'count'],
            'MONTHS_BALANCE': ['max', 'min', np.ptp],
            'CNT_INSTALMENT': ['min', 'max', 'mean', 'median', 'nunique', 'std'],
            'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'median', 'nunique', 'std'],
            'SK_DPD': ['sum', 'min', 'max', 'mean'],
            'SK_DPD_DEF': ['sum', 'min', 'max', 'mean'],
            'SK_DPD_diff': ['sum', 'min', 'max', 'mean'],
            'SK_DPD_rate': ['mean', 'sum'],
            'SK_DPD_DEF_rate': ['mean', 'sum'],
        }
        agg = pcb.groupby('SK_ID_CURR').agg(agg)
        agg.columns = [c[0] + '_' + c[1] for c in agg.columns.tolist()]
        data = data.merge(agg.reset_index(), on='SK_ID_CURR', how='left')

        stat = pcb[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
        agg = {
            'MONTHS_BALANCE': ['min', 'max', np.ptp],
            'CNT_INSTALMENT': ['min', 'max', 'nunique', np.ptp, 'median'],
            'CNT_INSTALMENT_FUTURE': ['max', 'min', 'mean', 'median', 'nunique', 'std'],
            'SK_DPD': ['max', 'min', np.ptp],
            'SK_DPD_DEF': ['max', 'min', np.ptp],
            'SK_DPD_diff': ['max', 'min', np.ptp],
        }
        agg = pcb.groupby('SK_ID_PREV').agg(agg)
        agg.columns = [c[0] + '_' + c[1] for c in agg.columns.tolist()]
        stat = stat.merge(agg.reset_index(), on='SK_ID_PREV', how='left')

        agg = {c: ['min', 'max', 'mean', 'sum'] for c in stat.columns if c not in ['SK_ID_CURR', 'SK_ID_PREV']}

        agg = stat.groupby('SK_ID_CURR').agg(agg)
        agg.columns = [c[0] + '_' + c[1] for c in agg.columns.tolist()]
        data = data.merge(agg.reset_index(), on='SK_ID_CURR', how='left')

        stat = pcb.sort_values('MONTHS_BALANCE', ascending='True').drop_duplicates('SK_ID_PREV', keep='last')
        stat = stat.groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).size().unstack().add_suffix(
            '_last_count').reset_index()
        data = data.merge(stat, on='SK_ID_CURR', how='left')
        data['Completed_last_count_ratio'] = data['Completed_last_count'] / data['SK_ID_PREV_nunique']
        data['Active_last_count_ratio'] = data['Active_last_count'] / data['SK_ID_PREV_nunique']

        stat = pcb.groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).size().unstack().add_suffix('_count').reset_index()
        data = data.merge(stat, on='SK_ID_CURR', how='left')
        data['Completed_last_count_ratio'] = data['Completed_last_count'] / data['SK_ID_PREV_count']
        data['Active_last_count_ratio'] = data['Active_last_count'] / data['SK_ID_PREV_count']

        # 提前还款次数
        stat = pcb.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT'].nunique().reset_index()
        data['SK_ID_PREV_advance_count'] = groupby(data, stat[stat['CNT_INSTALMENT'] > 1], 'SK_ID_CURR', 'SK_ID_CURR',
                                                   'count')
        data['SK_ID_PREV_ontime_count'] = data['SK_ID_PREV_nunique'] - data['SK_ID_PREV_advance_count']
        data['SK_ID_PREV_ontime_count_ratio'] = data['SK_ID_PREV_ontime_count'] / data['SK_ID_PREV_nunique']

        # 提前还款次数
        stat = stat[stat['SK_ID_PREV'].isin(pcb[pcb['NAME_CONTRACT_STATUS'] == 'Completed']['SK_ID_PREV'].unique())]
        data['SK_ID_PREV_nunique2'] = groupby(data, stat[stat['CNT_INSTALMENT'] > 1], 'SK_ID_CURR', 'SK_ID_PREV',
                                              'nunique')
        data['SK_ID_PREV_advance_count2'] = groupby(data, stat[stat['CNT_INSTALMENT'] > 1], 'SK_ID_CURR', 'SK_ID_CURR',
                                                    'count')
        data['SK_ID_PREV_ontime_count2'] = data['SK_ID_PREV_nunique2'] - data['SK_ID_PREV_advance_count2']
        data['SK_ID_PREV_ontime_count_ratio2'] = data['SK_ID_PREV_ontime_count2'] / data['SK_ID_PREV_nunique2']

        # 本月欠款次数
        stat = pcb[pcb['MONTHS_BALANCE'] == -1]
        stat['build'] = groupby(stat, pcb, 'SK_ID_PREV', 'MONTHS_BALANCE', 'min')
        stat['build'] = (stat['build'] == stat['MONTHS_BALANCE']).astype(int)
        data['SK_ID_PREV_month1_count'] = groupby(data, stat, 'SK_ID_CURR', 'SK_ID_PREV', 'nunique')
        data['SK_ID_PREV_month1_active_count'] = groupby(data, stat[stat['NAME_CONTRACT_STATUS'] == 'Active'],
                                                         'SK_ID_CURR', 'SK_ID_PREV', 'nunique')
        data['SK_ID_PREV_month1_Completed_count'] = groupby(data, stat[stat['NAME_CONTRACT_STATUS'] == 'Completed'],
                                                            'SK_ID_CURR', 'SK_ID_PREV', 'nunique')
        data['SK_ID_PREV_month1_active_count_ratio'] = data['SK_ID_PREV_month1_active_count'] / data[
            'SK_ID_PREV_month1_count']

        stat = pcb[pcb['SK_DPD'] > 0]
        data['SK_DPD_count'] = groupby(data, stat, 'SK_ID_CURR', 'SK_ID_PREV', 'nunique')
        stat = pcb[pcb['SK_DPD_DEF'] > 0]
        data['SK_DPD_DEF_count'] = groupby(data, stat, 'SK_ID_CURR', 'SK_ID_PREV', 'nunique')
        stat = pcb[pcb['SK_DPD_diff'] > 0]
        data['SK_DPD_diff_count'] = groupby(data, stat, 'SK_ID_CURR', 'SK_ID_PREV', 'nunique')

        for month in [54, 28, 13, 6, 3]:
            pcb = pcb[pcb['MONTHS_BALANCE'] > -month]
            pcb['SK_DPD_diff'] = pcb['SK_DPD'] - pcb['SK_DPD_DEF']
            agg = {
                'MONTHS_BALANCE': ['max', 'min', np.ptp],
                'CNT_INSTALMENT': ['std'],
                'CNT_INSTALMENT_FUTURE': ['mean'],
                'SK_DPD': ['mean'],
                'SK_DPD_DEF': ['mean'],
                'SK_DPD_rate': ['mean', 'sum'],
                'SK_DPD_DEF_rate': ['mean', 'sum'],
            }
            agg = pcb.groupby('SK_ID_CURR').agg(agg)
            agg.columns = [c[0] + '_' + c[1] for c in agg.columns.tolist()]
            agg.columns = [c + '_{}month'.format(month) for c in agg.columns]
            data = data.merge(agg.reset_index(), on='SK_ID_CURR', how='left')

            stat = pcb[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
            agg = {
                'MONTHS_BALANCE': ['min', 'max', np.ptp],
                'CNT_INSTALMENT': ['min', 'max', np.ptp],
                'CNT_INSTALMENT_FUTURE': ['min', 'std', 'mean', 'median', 'nunique'],
                'SK_DPD': ['max'],
                'SK_DPD_DEF': ['max'],
            }
            agg = pcb.groupby('SK_ID_PREV').agg(agg)
            agg.columns = [c[0] + '_' + c[1] for c in agg.columns.tolist()]
            stat = stat.merge(agg.reset_index(), on='SK_ID_PREV', how='left')

            # agg = {c:['min','max','mean','sum'] for c in stat.columns if c not in ['SK_ID_CURR','SK_ID_PREV']}
            agg = {
                'MONTHS_BALANCE_min': ['mean', 'sum', 'max'],
                'MONTHS_BALANCE_max': ['min', 'sum', 'max'],
                'CNT_INSTALMENT_ptp': ['mean', 'max'],
                'CNT_INSTALMENT_FUTURE_std': ['min', 'max', 'mean', 'sum'],
                'CNT_INSTALMENT_FUTURE_median': ['max', 'mean'],
                'CNT_INSTALMENT_FUTURE_min': ['mean', 'sum'],
                'CNT_INSTALMENT_FUTURE_mean': ['max'],
                'SK_DPD_DEF_max': ['mean'],
                'SK_DPD_max': ['mean'],
                'MONTHS_BALANCE_ptp': ['mean'],
                'CNT_INSTALMENT_min': ['mean'],
                'CNT_INSTALMENT_FUTURE_nunique': ['sum']
            }
            agg = stat.groupby('SK_ID_CURR').agg(agg)
            agg.columns = [c[0] + '_' + c[1] for c in agg.columns.tolist()]
            agg.columns = [c + '_{}month'.format(month) for c in agg.columns]
            data = data.merge(agg.reset_index(), on='SK_ID_CURR', how='left')

        data.columns = ['pcb_' + c if c not in ['SK_ID_CURR', 'label'] else c for c in data.columns]
        data.to_feather(result_path)
    return data

# POS_CASH_balance特征
def get_POS_CASH_balance_feat2(ids,data_key):
    result_path = cache_path + 'POS_CASH_balance2_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        pos = pd.read_csv(data_path + 'POS_CASH_balance.csv')
        pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
        # Features
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']

        pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        # Count pos cash accounts
        pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
        data = ids.merge(pos_agg.reset_index(),on='SK_ID_CURR',how='left')
        data.to_feather(result_path)
    return data

# previous_applications特征
def get_previous_applications_feat(ids,data_key):
    result_path = cache_path + 'previous_applications_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        data = ids.copy()
        prev = pd.read_csv(data_path + 'previous_application.csv').sort_values('DAYS_DECISION')
        prev['APP_CREDIT_rate'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        prev['APP_CREDIT_diff'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        prev['DAYS_DECISION_diff'] = prev.groupby('SK_ID_CURR')['DAYS_DECISION'].shift(1)
        prev['AMT_DOWN_PAYMENT_rate'] = prev['AMT_DOWN_PAYMENT'] / (prev['AMT_CREDIT'] + 0.01)
        prev['NFLAG_LAST_APPL_IN_DAY'] = 1 - prev['NFLAG_LAST_APPL_IN_DAY']
        prev['return_day'] = prev['DAYS_DECISION'] + prev['CNT_PAYMENT'] * 30
        prev['DAYS_TERMINATION_diff'] = prev['DAYS_TERMINATION'] - prev['return_day']
        prev['FLAG_LAST_APPL_PER_CONTRACT'] = prev['FLAG_LAST_APPL_PER_CONTRACT'].map({'Y': 0, 'N': 1})
        prev['DAYS_FIRST_DUE_diff'] = prev['DAYS_DECISION'] - prev['DAYS_FIRST_DUE']
        prev['DAYS_LAST_DUE_diff'] = prev['DAYS_DECISION'] - prev['DAYS_LAST_DUE']
        prev['DAYS_LAST_DUE_1ST_VERSION_diff'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']
        prev, cat_cols = one_hot_encoder(prev, nan_as_category=True, min_count=1000)
        # Previous applications numeric features
        num_aggregations = {
            'SK_ID_PREV': ['count'],
            'AMT_ANNUITY': ['min', 'max', 'mean', 'median', 'std', np.ptp, 'first', 'last'],
            'AMT_APPLICATION': ['min', 'max', 'mean', 'std'],
            'AMT_CREDIT': ['min', 'max', 'mean', 'std', 'sum'],
            'APP_CREDIT_rate': ['max', 'mean'],
            'APP_CREDIT_diff': ['min', 'max', 'mean', 'sum'],
            'AMT_DOWN_PAYMENT_rate': ['min', 'max', 'mean', 'std', np.ptp, 'first', 'last'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'std', 'first', 'last'],
            'AMT_GOODS_PRICE': ['max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['max', 'min', 'mean'],
            'FLAG_LAST_APPL_PER_CONTRACT': ['max', 'mean'],
            'NFLAG_LAST_APPL_IN_DAY': ['mean', 'sum'],
            'RATE_DOWN_PAYMENT': ['max', 'mean'],
            'RATE_INTEREST_PRIMARY': ['max', 'mean'],
            'RATE_INTEREST_PRIVILEGED': ['max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'median', 'mean', 'std', np.ptp],
            'DAYS_DECISION_diff': ['min', 'max', 'mean', 'std', 'last'],
            'DAYS_FIRST_DUE_diff': ['min', 'max', 'mean', 'std', 'last'],
            'DAYS_LAST_DUE_diff': ['min', 'max', 'mean', 'std', 'last'],
            'DAYS_LAST_DUE_1ST_VERSION_diff': ['min', 'max', 'mean', 'std', 'last'],
            'CNT_PAYMENT': ['min', 'max', 'mean', 'std', 'sum', 'last', 'first'],
            'return_day': ['min', 'max', 'mean', 'std', 'sum', 'last', 'first'],
            'SELLERPLACE_AREA': ['min', 'max', 'mean', 'last'],
            'DAYS_TERMINATION': ['min', 'max', 'mean', 'std', 'sum', 'last', 'first'],
            'DAYS_TERMINATION_diff': ['min', 'max', 'mean', 'std', 'sum', 'last', 'first'],
        }
        # Previous applications categorical features
        cat_aggregations = {c: ['mean', 'sum'] for c in cat_cols}
        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(
            ['PREV_' + e[0] + "_" + e[1] + '{}days'.format(3000) for e in prev_agg.columns.tolist()])
        data = data.merge(prev_agg.reset_index(), how='left', on='SK_ID_CURR')

        num_aggregations = {
            'AMT_ANNUITY': ['min', 'mean', 'median', 'first'],
            'APP_CREDIT_rate': ['mean'],
            'APP_CREDIT_diff': ['min'],
            'AMT_DOWN_PAYMENT': ['max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['mean'],
            'DAYS_DECISION': ['max'],
            'DAYS_DECISION_diff': ['max'],
            'CNT_PAYMENT': ['mean', 'std'],
            'return_day': ['max', 'std', 'sum'],
            'DAYS_TERMINATION': ['std', 'last'],
            'DAYS_TERMINATION_diff': ['min', 'max', 'std', 'first'],
            'DAYS_FIRST_DUE_diff': ['min', 'mean'],
            'DAYS_LAST_DUE_1ST_VERSION_diff': ['min'],
            'DAYS_LAST_DUE_diff': ['min'],
            'NAME_YIELD_GROUP_high': ['mean'],
            'NAME_YIELD_GROUP_low_action': ['mean'],
            'PRODUCT_COMBINATION_Cash X-Sell: low': ['mean'],
            'NAME_CLIENT_TYPE_New': ['mean'],
            'NAME_CONTRACT_STATUS_Refused': ['mean'],
            'PRODUCT_COMBINATION_Cash X-Sell: high': ['mean']
        }
        for days in [1300, 580, 280, 90]:
            prev = prev[prev['DAYS_DECISION'] > -days].copy()
            prev_agg = prev.groupby('SK_ID_CURR').agg(num_aggregations)
            prev_agg.columns = pd.Index(
                ['PREV_' + e[0] + "_" + e[1] + '{}days'.format(days) for e in prev_agg.columns.tolist()])
            data = data.merge(prev_agg.reset_index(), how='left', on='SK_ID_CURR')
        data.columns = ['prev_' + c if c not in ['SK_ID_CURR', 'label'] else c for c in data.columns]
        data.to_feather(result_path)
    return data

# previous_applications特征
def get_previous_applications_feat2(ids,data_key):
    result_path = cache_path + 'previous_applications2_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        data = ids.copy()
        prev = pd.read_csv(data_path + 'previous_application.csv')
        prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['max', 'mean'],
            'AMT_APPLICATION': ['max', 'mean'],
            'AMT_CREDIT': ['max', 'mean'],
            'APP_CREDIT_PERC': ['max', 'mean'],
            'AMT_DOWN_PAYMENT': ['max', 'mean'],
            'AMT_GOODS_PRICE': ['max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['max', 'mean'],
            'RATE_DOWN_PAYMENT': ['max', 'mean'],
            'DAYS_DECISION': ['max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }
        # Previous applications categorical features
        cat_aggregations = {c:['mean','sum'] for c in cat_cols}
        for days in [3000,500,200,100]:
            prev = prev[prev['DAYS_DECISION']>-days].copy()
            prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
            prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1] + '{}days'.format(days) for e in prev_agg.columns.tolist()])
        # # Previous Applications: Approved Applications - only numerical features
        # approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        # approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        # approved_agg.columns = pd.Index(
        #     ['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        # prev_agg = prev_agg.join(approved_agg, how='left')
        # # Previous Applications: Refused Applications - only numerical features
        # refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        # refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        # refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        # prev_agg = prev_agg.join(refused_agg, how='left')
            data = data.merge(prev_agg.reset_index(), how='left', on='SK_ID_CURR')

        data.to_feather(result_path)
    return data


# credit_card_balance特征
def get_credit_card_balance_feat(ids, data_key):
    result_path = cache_path + 'credit_card_balance_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        cc = pd.read_csv(data_path + 'credit_card_balance.csv').sort_values('MONTHS_BALANCE')
        cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
        cc['AMT_BALANCE_int'] = (cc['AMT_BALANCE'] > 0).astype(int)
        cc['AMT_BALANCE_diff'] = cc['AMT_BALANCE'] - cc.groupby('SK_ID_PREV')['AMT_BALANCE'].shift(1)
        cc['AMT_INST_MIN_REGULARITY_diff'] = cc['AMT_INST_MIN_REGULARITY'] - cc['AMT_PAYMENT_CURRENT']
        cc['AMT_INST_MIN_REGULARITY_rate'] = cc['AMT_INST_MIN_REGULARITY'] / cc['AMT_PAYMENT_CURRENT']
        cc['AMT_INST_MIN_REGULARITY_diff1'] = cc['AMT_PAYMENT_TOTAL_CURRENT'] - cc['AMT_PAYMENT_CURRENT']
        cc['AMT_INST_MIN_REGULARITY_diff2'] = cc['AMT_PAYMENT_TOTAL_CURRENT'] - cc['AMT_INST_MIN_REGULARITY']
        cc_agg = {'SK_ID_PREV': ['nunique', 'count'],
                  'MONTHS_BALANCE': ['max', 'min', 'std'],
                  'AMT_BALANCE': ['min', 'max', 'mean', 'std', 'first', 'last', np.ptp],
                  'AMT_BALANCE_diff': ['min', 'max', 'mean', 'std', 'last', 'first'],
                  'AMT_BALANCE_int': ['mean', 'sum'],
                  'AMT_DRAWINGS_CURRENT': ['sum'],
                  'AMT_DRAWINGS_ATM_CURRENT': ['sum'],
                  'AMT_DRAWINGS_POS_CURRENT': ['sum'],
                  'AMT_DRAWINGS_OTHER_CURRENT': ['sum'],
                  'AMT_INST_MIN_REGULARITY': ['max', 'mean', 'std', 'sum', 'last'],
                  'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'std', 'sum', 'first', 'last'],
                  'AMT_INST_MIN_REGULARITY_diff': ['min', 'max', 'mean', 'std', 'sum', 'first', 'last'],
                  'AMT_INST_MIN_REGULARITY_rate': ['min', 'max', 'mean', 'std', 'first', 'last'],
                  'AMT_INST_MIN_REGULARITY_diff1': ['min', 'mean', 'std', 'sum', 'last'],
                  'AMT_INST_MIN_REGULARITY_diff2': ['min', 'max', 'mean', 'std', 'sum', 'first', 'last'],
                  'AMT_RECEIVABLE_PRINCIPAL': ['sum'],
                  'AMT_RECIVABLE': ['sum'],
                  'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
                  'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
                  'CNT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum'],
                  'CNT_INSTALMENT_MATURE_CUM': ['max', 'last', 'sum'],
                  'CNT_DRAWINGS_OTHER_CURRENT': ['sum'],
                  'NAME_CONTRACT_STATUS_Active': ['mean', 'sum'],
                  'NAME_CONTRACT_STATUS_Completed': ['sum'],
                  }

        cc_agg = cc.groupby('SK_ID_CURR').agg(cc_agg)
        cc_agg.columns = pd.Index([e[0] + "_" + e[1] for e in cc_agg.columns.tolist()])
        data = ids.merge(cc_agg.reset_index(), how='left', on='SK_ID_CURR')

        data['AMT_DRAWINGS_ATM_CURRENT_sum_ratio'] = data['AMT_DRAWINGS_ATM_CURRENT_sum'] / (
                    data['AMT_DRAWINGS_CURRENT_sum'] + 0.01)
        data['AMT_DRAWINGS_POS_CURRENT_sum_ratio'] = data['AMT_DRAWINGS_POS_CURRENT_sum'] / (
                    data['AMT_DRAWINGS_CURRENT_sum'] + 0.01)
        data['AMT_DRAWINGS_OTHER_CURRENT_sum_ratio'] = data['AMT_DRAWINGS_OTHER_CURRENT_sum'] / (
                    data['AMT_DRAWINGS_CURRENT_sum'] + 0.01)
        data['AMT_RECIVABLE_sum_ratio'] = data['AMT_RECIVABLE_sum'] / (data['AMT_DRAWINGS_CURRENT_sum'] + 0.01)
        data['ACNT_DRAWINGS_ATM_CURRENT_sum_ratio'] = data['CNT_DRAWINGS_ATM_CURRENT_sum'] / (
                    data['CNT_DRAWINGS_CURRENT_sum'] + 0.01)
        data['CNT_DRAWINGS_POS_CURRENT_sum_ratio'] = data['CNT_DRAWINGS_POS_CURRENT_sum'] / (
                    data['CNT_DRAWINGS_CURRENT_sum'] + 0.01)
        data['CNT_DRAWINGS_OTHER_CURRENT_sum_ratio'] = data['CNT_DRAWINGS_OTHER_CURRENT_sum'] / (
                    data['CNT_DRAWINGS_CURRENT_sum'] + 0.01)
        stat = cc.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()
        data['CNT_INSTALMENT_MATURE_CUM_sum'] = groupby(data, stat, 'SK_ID_CURR', 'CNT_INSTALMENT_MATURE_CUM', 'sum')

        stat = cc.groupby(['SK_ID_CURR', 'SK_ID_PREV']).apply(
            lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()
        data['credit_card_avg_loading_of_credit_limit'] = groupby(data, stat, 'SK_ID_CURR', 0, 'mean')
        data['credit_card_avg_loading_of_credit_limit_mean'] = data['credit_card_avg_loading_of_credit_limit'] / data[
            'SK_ID_PREV_nunique']

        data.columns = [c + '_100month' if ((c not in ['SK_ID_CURR', 'label']) and ('month' not in c)) else c for c in
                        data.columns]

        cc_agg = {'SK_ID_PREV': ['nunique'],
                  'AMT_BALANCE': ['min', 'max', 'mean', 'std', 'first', np.ptp],
                  'AMT_BALANCE_diff': ['mean', 'std', 'first'],
                  'AMT_BALANCE_int': ['mean'],
                  'AMT_RECIVABLE': ['sum'],
                  'AMT_DRAWINGS_CURRENT': ['sum'],
                  'AMT_DRAWINGS_POS_CURRENT': ['sum'],
                  'AMT_INST_MIN_REGULARITY': ['std'],
                  'AMT_PAYMENT_CURRENT': ['sum', 'first'],
                  'AMT_INST_MIN_REGULARITY_diff': ['sum', 'first'],
                  'AMT_INST_MIN_REGULARITY_rate': ['mean', 'std', 'first'],
                  'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
                  'CNT_DRAWINGS_ATM_CURRENT': ['mean'],
                  'NAME_CONTRACT_STATUS_Active': ['sum'],
                  }
        for month in [55, 28, 12, 6, 3]:
            cc = cc[cc['MONTHS_BALANCE'] >= -month]
            aggregations = cc.groupby('SK_ID_CURR').agg(cc_agg)
            aggregations.columns = pd.Index([e[0] + "_" + e[1] for e in aggregations.columns.tolist()])
            data = data.merge(aggregations.reset_index(), how='left', on='SK_ID_CURR')

            data['AMT_RECIVABLE_sum_ratio'] = data['AMT_RECIVABLE_sum'] / (data['AMT_DRAWINGS_CURRENT_sum'] + 0.01)
            stat = cc.groupby(['SK_ID_CURR', 'SK_ID_PREV']).apply(
                lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()
            data['credit_card_avg_loading_of_credit_limit'] = groupby(data, stat, 'SK_ID_CURR', 0, 'mean')
            data['credit_card_avg_loading_of_credit_limit_mean'] = data['credit_card_avg_loading_of_credit_limit'] / \
                                                                   data['SK_ID_PREV_nunique']
            data.columns = [
                c + '_{}month'.format(month) if ((c not in ['SK_ID_CURR', 'label']) and ('month' not in c)) else c for c
                in data.columns]
        data.columns = ['ccb_' + c if c not in ['SK_ID_CURR', 'label'] else c for c in data.columns]
        data.to_feather(result_path)
    return data

# credit_card_balance特征
def get_credit_card_balance_feat2(ids, data_key):
    result_path = cache_path + 'credit_card_balance2_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        cc = pd.read_csv(data_path + 'credit_card_balance.csv').sort_values('MONTHS_BALANCE')
        cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
        cc['AMT_BALANCE_int'] = (cc['AMT_BALANCE']>0).astype(int)
        cc['AMT_BALANCE_diff'] = cc['AMT_BALANCE'] - cc.groupby('SK_ID_PREV')['AMT_BALANCE'].shift(1)
        cc_agg = {c :['max', 'mean', 'sum', np.ptp, 'std', 'last', 'first'] for c in cc.columns if c not in cat_cols}
        cc_agg.update({c :['mean', 'sum','last', 'first'] for c in cat_cols})

        cc_agg['SK_ID_PREV'] = ['count','nunique']
        cc_agg['AMT_BALANCE_int'] =  ['mean']
        cc_agg['AMT_BALANCE_diff'] = ['max', 'mean', 'sum', np.ptp, 'std', 'last', 'first']
        # cc_agg[''] = ['']
        # cc_agg[''] = ['']
        # cc_agg[''] = ['']
        # cc_agg[''] = ['']
        # cc_agg[''] = ['']
        # cc_agg[''] = ['']


        cc_agg = cc.groupby('SK_ID_CURR').agg(cc_agg)
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1] for e in cc_agg.columns.tolist()])
        # Count credit card lines
        data = ids.merge(cc_agg.reset_index(), how='left', on='SK_ID_CURR')

        data['AMT_DRAWINGS_ATM_CURRENT_sum_ratio'] = data['CC_AMT_DRAWINGS_ATM_CURRENT_sum'] / (
                    data['CC_AMT_DRAWINGS_CURRENT_sum'] + 0.01)
        stat = cc.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()
        data['CNT_INSTALMENT_MATURE_CUM_sum'] = groupby(data, stat, 'SK_ID_CURR', 'CNT_INSTALMENT_MATURE_CUM', 'sum')

        stat = cc.groupby(['SK_ID_CURR', 'SK_ID_PREV']).apply(
            lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()
        data['credit_card_avg_loading_of_credit_limit'] = groupby(data, stat, 'SK_ID_CURR', 0, 'mean')
        data['credit_card_avg_loading_of_credit_limit_mean'] = data['credit_card_avg_loading_of_credit_limit']/data['CC_SK_ID_PREV_nunique']

        # data = ids.copy()
        # cc = pd.read_csv(data_path + 'credit_card_balance.csv')
        # cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
        # # General aggregations
        # cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
        # cc1 = cc[[c for c in cc.columns if c not in cat_cols]]
        # cc2 = cc[[c for c in cc.columns if c in cat_cols]]
        # cc_agg1 = cc1.groupby('SK_ID_CURR').agg(['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first',trend])
        # cc_agg1.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg1.columns.tolist()])
        # cc_agg2 = cc2.groupby('SK_ID_CURR').agg(['mean', 'sum', trend])
        # cc_agg2.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg2.columns.tolist()])
        # # Count credit card lines
        # cc_agg1['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        # data = data.merge(cc_agg1.reset_index(),on='SK_ID_CURR',how='left')
        # data = data.merge(cc_agg2.reset_index(), on='SK_ID_CURR', how='left')

        # cc = pd.read_csv(data_path + 'credit_card_balance.csv').sort_values('MONTHS_BALANCE')
        # data = ids.copy()
        # aggregations = {
        #     'SK_ID_PREV': ['count', 'nunique'],
        #     'MONTHS_BALANCE': ['min','max', 'sum', 'last','first'],
        #     'AMT_BALANCE': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first',trend],
        #     'AMT_CREDIT_LIMIT_ACTUAL': ['min','max', 'mean', 'last','first',trend],
        #     'AMT_DRAWINGS_ATM_CURRENT': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first'],
        #     'AMT_DRAWINGS_CURRENT': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first'],
        #     'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'mean', 'sum', 'std','first'],
        #     'AMT_DRAWINGS_POS_CURRENT': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first'],
        #     'AMT_INST_MIN_REGULARITY': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first'],
        #     'AMT_PAYMENT_CURRENT': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first'],
        #     'AMT_PAYMENT_TOTAL_CURRENT': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first'],
        #     'AMT_RECEIVABLE_PRINCIPAL': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first'],
        #     'AMT_RECIVABLE': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last'],
        #     'AMT_TOTAL_RECEIVABLE': ['min','max', 'mean', 'sum',np.ptp, 'last','first'],
        #     'CNT_DRAWINGS_OTHER_CURRENT': ['min','mean'],
        #     'CNT_DRAWINGS_POS_CURRENT': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first'],
        #     'CNT_INSTALMENT_MATURE_CUM': ['min','max', 'mean', 'sum',np.ptp, 'std', 'last','first'],
        #     'SK_DPD': ['min','mean', 'std','first'],
        #     'SK_DPD_DEF': ['min','max', 'mean', 'sum',np.ptp, 'std']
        # }
        # for month in [100,50,25,12,6,3]:
        #     cc = cc[cc['MONTHS_BALANCE']>=-month]
        #     cc_agg = cc.groupby('SK_ID_CURR').agg(aggregations)
        #     cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1] for e in cc_agg.columns.tolist()])
        #     cc_agg.columns = [c + '_{}months'.format(month) for c in cc_agg.columns]
        #     data = data.merge(cc_agg.reset_index(), how='left', on='SK_ID_CURR')
        #
        #     stat = cc[cc['NAME_CONTRACT_STATUS'].isin(['Active', 'Completed', 'Signed', 'Demand'])].groupby(
        #         ['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).size().unstack().add_suffix('_count').add_prefix(
        #         '{}_'.format('NAME_CONTRACT_STATUS'))
        #     stat.columns = [c + '_{}months'.format(month) for c in stat.columns]
        #     data = data.merge(stat.reset_index(), on='SK_ID_CURR', how='left')
        #     for c in stat.columns:
        #         data[c + '_ratio'] = data[c] / data['CC_SK_ID_PREV_count'+ '_{}months'.format(month)]

        data.columns = ['cc_' + c if c not in ['SK_ID_CURR', 'label'] else c for c in data.columns]
        data.to_feather(result_path)
    return data


# installments_payments特征
def get_installments_payments_feat(ids, data_key):
    result_path = cache_path + 'installments_payments_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feather(result_path)
    else:
        data = ids.copy()
        ins = pd.read_csv(data_path + 'installments_payments.csv').sort_values('DAYS_INSTALMENT')
        ins['DAYS_INSTALMENT_diff'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DAYS_INSTALMENT_diff2'] = (ins['DAYS_INSTALMENT_diff']>0).astype(int)
        ins['AMT_PAYMENT_diff'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
        ins['AMT_PAYMENT_diff2'] = (ins['AMT_PAYMENT_diff'] >= 0).astype(int)

        aggregations = {
            'SK_ID_PREV': ['nunique','count'],
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DAYS_INSTALMENT_diff': ['max', 'min', 'mean','median', 'std'],
            'DAYS_INSTALMENT_diff2': ['mean'],
            'AMT_PAYMENT_diff': ['max', 'mean', 'std', 'min'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min', 'std'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'std']
        }
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1] for e in ins_agg.columns.tolist()])
        data = data.merge(ins_agg.reset_index(),on='SK_ID_CURR',how='left')

        data['AMT_PAYMENT_sum_diff'] = data['INSTAL_AMT_INSTALMENT_sum'] - data['INSTAL_AMT_PAYMENT_sum']

        stat = ins.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].max().reset_index()
        data['INSTAL_NUM_INSTALMENT_NUMBER_max_min'] = groupby(data,stat,'SK_ID_CURR','NUM_INSTALMENT_NUMBER','min')
        data['INSTAL_NUM_INSTALMENT_NUMBER_max_mean'] = groupby(data, stat, 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER', 'mean')


        stat = ins[ins['NUM_INSTALMENT_VERSION']<6].groupby(['SK_ID_CURR', 'NUM_INSTALMENT_VERSION']).size().unstack().add_suffix(
            '_count').add_prefix('INSTAL_NUM_INSTALMENT_VERSION_')
        data = data.merge(stat.reset_index(), on='SK_ID_CURR', how='left')
        for c in stat.columns:
            data[c + '_ratio'] = data[c] / data['INSTAL_SK_ID_PREV_count']

        stat = ins.groupby(['SK_ID_CURR', 'DAYS_INSTALMENT_diff2']).size().unstack().add_suffix(
            '_count').add_prefix('INSTAL_DAYS_INSTALMENT_diff2_')
        data = data.merge(stat.reset_index(), on='SK_ID_CURR', how='left')
        for c in stat.columns:
            data[c + '_ratio'] = data[c] / data['INSTAL_SK_ID_PREV_count']

        stat = ins.groupby(['SK_ID_CURR', 'AMT_PAYMENT_diff2']).size().unstack().add_suffix(
            '_count').add_prefix('INSTAL_AMT_PAYMENT_diff2_')
        data = data.merge(stat.reset_index(), on='SK_ID_CURR', how='left')
        for c in stat.columns:
            data[c + '_ratio'] = data[c] / data['INSTAL_SK_ID_PREV_count']

        # 最后一次表现
        for i in range(3):
            stat = ins.groupby('SK_ID_CURR').shift(i).reset_index()
            stat['SK_ID_CURR'] = ins['SK_ID_CURR']
            stat = ins.groupby('SK_ID_CURR').last().drop('SK_ID_PREV', axis=1).add_suffix('_last_time{}'.format(i))
            data = data.merge(stat.reset_index(), on='SK_ID_CURR', how='left')
            stat = ins.groupby('SK_ID_CURR').shift(-i).reset_index()
            stat['SK_ID_CURR'] = ins['SK_ID_CURR']
            stat = ins.groupby('SK_ID_CURR').first().drop('SK_ID_PREV', axis=1).add_suffix('_first_time{}'.format(i))
            data = data.merge(stat.reset_index(), on='SK_ID_CURR', how='left')

        for days in [1461,731,366,125,63,32]:
            data2 = ids[['SK_ID_CURR']].copy()
            ins = ins[ins['DAYS_ENTRY_PAYMENT']>-days]
            ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
            ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1] for e in ins_agg.columns.tolist()])
            data2 = data2.merge(ins_agg.reset_index(), on='SK_ID_CURR', how='left')

            data2['AMT_PAYMENT_sum_diff'] = data2['INSTAL_AMT_INSTALMENT_sum'] - data2['INSTAL_AMT_PAYMENT_sum']

            stat = ins.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].max().reset_index()
            data2['INSTAL_NUM_INSTALMENT_NUMBER_max_min'] = groupby(data2, stat, 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER',
                                                                   'min')
            data2['INSTAL_NUM_INSTALMENT_NUMBER_max_mean'] = groupby(data2, stat, 'SK_ID_CURR', 'NUM_INSTALMENT_NUMBER',
                                                                    'mean')

            stat = ins[ins['NUM_INSTALMENT_VERSION'] < 6].groupby(
                ['SK_ID_CURR', 'NUM_INSTALMENT_VERSION']).size().unstack().add_suffix(
                '_count').add_prefix('INSTAL_NUM_INSTALMENT_VERSION_')
            data2 = data2.merge(stat.reset_index(), on='SK_ID_CURR', how='left')
            for c in stat.columns:
                data2[c + '_ratio'] = data2[c] / data2['INSTAL_SK_ID_PREV_count']

            stat = ins.groupby(['SK_ID_CURR', 'DAYS_INSTALMENT_diff2']).size().unstack().add_suffix(
                '_count').add_prefix('INSTAL_DAYS_INSTALMENT_diff2_')
            data2 = data2.merge(stat.reset_index(), on='SK_ID_CURR', how='left')
            for c in stat.columns:
                data2[c + '_ratio'] = data2[c] / data2['INSTAL_SK_ID_PREV_count']

            stat = ins.groupby(['SK_ID_CURR', 'AMT_PAYMENT_diff2']).size().unstack().add_suffix(
                '_count').add_prefix('INSTAL_AMT_PAYMENT_diff2_')
            data2 = data2.merge(stat.reset_index(), on='SK_ID_CURR', how='left')
            for c in stat.columns:
                data2[c + '_ratio'] = data2[c] / data2['INSTAL_SK_ID_PREV_count']
            data2.columns = [c + '_{}days'.format(days) for c in data2.columns]
            data = concat([data,data2])
        data.columns = ['ins_' + c if c not in ['SK_ID_CURR', 'label'] else c for c in data.columns]
        data.to_feather(result_path)
    return data

# installments_payments特征
def get_installments_payments_feat2(ids, data_key):
    result_path = cache_path + 'installments_payments2_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 0:
        data = pd.read_feather(result_path)
    else:
        # One-hot encoding for categorical columns with get_dummies
        def one_hot_encoder(df, nan_as_category=True):
            original_columns = list(df.columns)
            categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
            df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
            new_columns = [c for c in df.columns if c not in original_columns]
            return df, new_columns

        ins = pd.read_csv(data_path + 'installments_payments.csv')
        ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
        # Percentage and difference paid in each installment (amount paid and installment value)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        # Days past due and days before due (no negative values)
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum', 'min', 'std'],
            'DBD': ['max', 'mean', 'sum', 'min', 'std'],
            'PAYMENT_PERC': ['max', 'mean', 'std', 'min'],
            'PAYMENT_DIFF': ['max', 'mean', 'std', 'min'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min', 'std'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'std']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        # Count installments accounts


        data = ids.merge(ins_agg.reset_index(), how='left', on='SK_ID_CURR')
        data.to_feather(result_path)
    return data


# 二次处理特征
def second_feat(result):
    # previous_applications/ pos_cash /  bureau
    return result

def make_feat(data,data_key):
    t0 = time.time()
    result_path = cache_path + 'feat_set_{}.feather'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_feather(result_path, 'w')
    else:
        data = get_attribute(data,data_key)
        # data = get_attribute2(data, data_key)
        ids = data[['SK_ID_CURR','label']]

        result = [data]
        print('开始构造特征...')
        result.append(get_bureau_and_balance_feat(ids,data_key))                    # bureau特征
        # result.append(get_bureau_and_balance_feat2(ids, data_key))                   # bureau特征
        result.append(get_POS_CASH_balance_feat(ids,data_key))                      # POS_CASH_balance特征
        # result.append(get_POS_CASH_balance_feat2(ids, data_key))                    # POS_CASH_balance特征
        result.append(get_previous_applications_feat(ids, data_key))                   # previous_applications特征
        # result.append(get_previous_applications_feat2(ids,data_key))                     # previous_applications特征
        result.append(get_credit_card_balance_feat(ids, data_key))                      # credit_card_balance特征
        # result.append(get_credit_card_balance_feat2(ids, data_key))                      # credit_card_balance特征
        result.append(get_installments_payments_feat(ids, data_key))                       # installments_payments
        # result.append(get_installments_payments_feat2(ids, data_key))                    # installments_payments
        print('开始合并特征...')
        result = concat(result)

        result = second_feat(result)
        # print('存储数据...')
        # result.to_feather(result_path)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result

