import lightgbm as lgb
from tool.tool import *
from credit.feat3 import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 尝试分组标准化
# 从时间序列的角度考虑  滑窗
# 增加样本
# 第一次最后一次
# 先统计细分类prev 再统计curr


cache_path = 'F:/credit/'
data_path = 'C:/Users/csw/Desktop/python/credit/data/'

cate_feat = ['CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','ORGANIZATION_TYPE']


train = pd.read_csv(data_path + 'application_train.csv').rename(columns = {'TARGET':'label'})
test_id = train.sample(frac=0.5,random_state=66)['SK_ID_CURR'].values
train_id = list(set(train['SK_ID_CURR'].values) - set(test_id))
pickle.dump(test_id,open(data_path + 'test_id.pkl','+wb'))
pickle.dump(train_id,open(data_path + 'train_id.pkl','+wb'))
test_id = pickle.load(open(data_path + 'test_id.pkl','+rb'))
train_id = pickle.load(open(data_path + 'train_id.pkl','+rb'))
test_y = train[train['SK_ID_CURR'].isin(test_id)]['label'].copy()
train.loc[train['SK_ID_CURR'].isin(test_id),'label'] = np.nan


data = make_feat(train,'offline')
# data = compress(data)

test_feat = data[data['SK_ID_CURR'].isin(test_id)].copy()
train_feat = data[~data['SK_ID_CURR'].isin(test_id)].copy()



predictors = [c for c in train_feat.columns if c not in ['label']]

t0 = time.time()
print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 8,
    'num_leaves': 32,
    'learning_rate': 0.02,
    'subsample': 0.8715623,
    'colsample_bytree': 0.9497036,
    'reg_alpha': 0.04,
    'reg_lambda': 0.073,
    'min_split_gain': 0.0222415,
    'min_child_weight':40,
    'verbose': -1,
    'seed': 66,
}
lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_test = lgb.Dataset(test_feat[predictors], test_y,reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval = 50,
                early_stopping_rounds=100)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')
preds = gbm.predict(test_feat[predictors])
print('线下得分：{}'.format(roc_auc_score(test_y,preds)))
print('CV训练用时{}秒'.format(time.time() - t0))
