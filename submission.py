from tool.tool import *
from credit.feat1 import *
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 尝试分组标准化
# 从时间序列的角度考虑  滑窗
# 增加样本
# 第一次最后一次
# 先统计细分类prev 再统计curr


cache_path = 'E:/credit/'
data_path = 'C:/Users/cui/Desktop/python/credit/data/'

cate_feat = ['CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','ORGANIZATION_TYPE']


train = pd.read_csv(data_path + 'application_train.csv').rename(columns = {'TARGET':'label'})
test = pd.read_csv(data_path + 'application_test.csv').rename(columns = {'TARGET':'label'})
data = train.append(test)

data = make_feat(data,'online')

folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=666)
# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
predictors = [c for c in data.columns if c not in ['label','SK_ID_CURR','index']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[predictors], train_df['label'])):
    train_x, train_y = train_df[predictors].iloc[train_idx], train_df['label'].iloc[train_idx]
    valid_x, valid_y = train_df[predictors].iloc[valid_idx], train_df['label'].iloc[valid_idx]

    # LightGBM parameters found by Bayesian optimization
    clf = LGBMClassifier(
        # is_unbalance=True,
        n_estimators=10000,
        learning_rate=0.01,
        num_leaves=32,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.04,
        reg_lambda=0.073,
        min_split_gain=0.0222415,
        min_child_weight=40,
        silent=-1,
        verbose=-1,
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric='auc', verbose=50, early_stopping_rounds=200)

    oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test_df[predictors], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del clf, train_x, train_y, valid_x, valid_y
    gc.collect()

print('Full AUC score %.6f' % roc_auc_score(train_df['label'], oof_preds))
# Write submission file and plot feature importance
test_df['TARGET'] = sub_preds
submission_file_name = r'C:\Users\cui\Desktop\python\credit\submission\sub{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)

name = 'all'
train_df['pred'] = oof_preds
all_pred = pd.concat([test_df[['SK_ID_CURR','TARGET']].rename(columns={'TARGET':'pred'}), train_df[['SK_ID_CURR','pred']]])
all_pred.rename(columns={'pred':'piupiu_{}_pred'.format(name)},inplace=True)
all_pred.to_csv(r'C:\Users\cui\Desktop\python\credit\submission\piupiu_lgb_{}_v1.0__valid{}.csv'.format(name,
    roc_auc_score(train_df['label'], oof_preds)),index=False)
