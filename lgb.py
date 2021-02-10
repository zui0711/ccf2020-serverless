import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import linregress


# 官方自定义评价函数
def get_score(y, y_pred, name):
    if name == 'CPU':
        t = 0.9*np.abs(y-y_pred)/100.
        return np.mean(t)
    else:
        max_v = np.max([y, y_pred], axis=0)
        max_v[max_v == 0.] = 1
        t = 0.1*np.divide(np.abs(y-y_pred), max_v)
        return np.mean(t)


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/evaluation_public.csv')

cpu_feats = ['CU', 'STATUS', 'QUEUE_TYPE', 'PLATFORM', 'QUEUE_ID',
             'CPU_USAGE', 'MEM_USAGE',
             'LAUNCHING_JOB_NUMS', 'RUNNING_JOB_NUMS',
             'RESOURCE_TYPE', 'DISK_USAGE',
             'TIME_HOUR']


# 构造label
def make_label(data):
    data['CPU_USAGE_1'] = data.CPU_USAGE.shift(-1)
    data['CPU_USAGE_2'] = data.CPU_USAGE.shift(-2)
    data['CPU_USAGE_3'] = data.CPU_USAGE.shift(-3)
    data['CPU_USAGE_4'] = data.CPU_USAGE.shift(-4)
    data['CPU_USAGE_5'] = data.CPU_USAGE.shift(-5)

    data['LAUNCHING_JOB_NUMS_1'] = data.LAUNCHING_JOB_NUMS.shift(-1)
    data['LAUNCHING_JOB_NUMS_2'] = data.LAUNCHING_JOB_NUMS.shift(-2)
    data['LAUNCHING_JOB_NUMS_3'] = data.LAUNCHING_JOB_NUMS.shift(-3)
    data['LAUNCHING_JOB_NUMS_4'] = data.LAUNCHING_JOB_NUMS.shift(-4)
    data['LAUNCHING_JOB_NUMS_5'] = data.LAUNCHING_JOB_NUMS.shift(-5)

    return data.dropna()


# 处理时间
def proc_time(df):
    df['DOTTING_TIME'] /= 1000
    df['DOTTING_TIME'] = list(map(
        lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)), df['DOTTING_TIME']))
    df = df.sort_values(['QUEUE_ID', 'DOTTING_TIME'])
    df['DOTTING_TIME'] = pd.to_datetime(df['DOTTING_TIME'])
    df['TIME_HOUR'] = df['DOTTING_TIME'].map(lambda x: x.hour)
    return df


df_train = proc_time(df_train)
df_test = proc_time(df_test)

# 数值变换及交叉特征
for df in [df_train, df_test]:
    df['CPU_USAGE'] = 10*np.sqrt(df['CPU_USAGE'])
    df['MEM_USAGE'] = 10*np.sqrt(df['MEM_USAGE'])
    df['DISK_USAGE'] = 10*np.sqrt(df['DISK_USAGE'])
    df['CU_CPU'] = df['CU'] * df['CPU_USAGE'] / 100.
    df['CU_MEM'] = df['CU'] * 4 * df['MEM_USAGE'] / 100.
    df['TO_DO_JOB'] = df['LAUNCHING_JOB_NUMS'] - df['RUNNING_JOB_NUMS']
    df['MEM_DISK'] = df['MEM_USAGE'] + df['DISK_USAGE']

cpu_feats = cpu_feats + ['CU_CPU', 'CU_MEM', 'TO_DO_JOB', 'MEM_DISK']

# 时序特征
for name in ['CPU_USAGE', 'MEM_USAGE', 'CU_CPU', 'MEM_DISK']:
    f = [name]
    # 多个均值
    for n in range(1,5):
        df_train[name+'_%d_ago'%n] = df_train[name].shift(n)
        df_test[name+'_%d_ago'%n] = df_test[name].shift(n)
        cpu_feats.append(name+'_%d_ago'%n)
        f.append(name+'_%d_ago'%n)

        df_train[name + '_mean_%d'%n] = df_train[f].mean(axis=1)
        df_test[name + '_mean_%d'%n] = df_test[f].mean(axis=1)
        cpu_feats.append(name + '_mean_%d'%n)
    # 趋势值
    df_train[name+'_0_trade'] = np.subtract(df_train[name], df_train[name+'_mean_4'])
    df_test[name+'_0_trade'] = np.subtract(df_test[name], df_test[name+'_mean_4'])
    cpu_feats.append(name+'_0_trade')
    for n in range(1, 5):
        df_train[name + '_%d_ago_trade' % n] = np.subtract(df_train[name+'_%d_ago'%n], df_train[name+'_mean_4'])
        df_test[name + '_%d_ago_trade' % n] = np.subtract(df_test[name+'_%d_ago'%n], df_test[name+'_mean_4'])
        cpu_feats.append(name + '_%d_ago_trade' % n)

for name in ['CPU_USAGE', 'CU_CPU']:
    for d in range(1,4):
        df_train[name + '_mean_%d_ratio'%d] = np.divide(df_train[name + '_mean_%d'%d]+1,
                                                        df_train[name + '_mean_4']+1)
        df_test[name + '_mean_%d_ratio'%d] = np.divide(df_test[name + '_mean_%d'%d]+1,
                                                       df_test[name + '_mean_4']+1)

    cpu_feats = cpu_feats + [name + '_mean_1_ratio', name + '_mean_2_ratio', name + '_mean_3_ratio']

# job类时序特征
for name in ['RUNNING_JOB_NUMS', 'TO_DO_JOB']:
    f = [name]
    for n in range(1, 5):
        df_train[name + '_%d_ago' % n] = df_train[name].shift(n)
        df_test[name + '_%d_ago' % n] = df_test[name].shift(n)
        cpu_feats.append(name + '_%d_ago' % n)
        f.append(name + '_%d_ago' % n)

    df_train[name + '_mean'] = df_train[f].mean(axis=1)
    df_test[name + '_mean'] = df_test[f].mean(axis=1)
    cpu_feats.append(name + '_mean')

for name in ['RUNNING_JOB_NUMS', 'TO_DO_JOB']:
    df_train[name + '_0_trade'] = np.subtract(df_train[name], df_train[name + '_mean'])
    df_test[name + '_0_trade'] = np.subtract(df_test[name], df_test[name + '_mean'])
    if name == 'RUNNING_JOB_NUMS':  # or name == 'TO_DO_JOB':
        cpu_feats.append(name + '_0_trade')
    for n in range(1, 5):
        df_train[name + '_%d_ago_trade' % n] = np.subtract(df_train[name + '_%d_ago' % n], df_train[name + '_mean'])
        df_test[name + '_%d_ago_trade' % n] = np.subtract(df_test[name + '_%d_ago' % n], df_test[name + '_mean'])
        if name == 'RUNNING_JOB_NUMS' or name == 'TO_DO_JOB':
            cpu_feats.append(name + '_%d_ago_trade' % n)

print(df_train.shape)

# 差分
for name in ['CPU_USAGE', 'MEM_USAGE', 'CU_CPU', 'MEM_DISK']:
    df_train[name+'_diff'] = df_train[name].diff()
    df_test[name+'_diff'] = df_test[name].diff()
    cpu_feats.append(name+'_diff')
    for d in range(1, 4):
        df_train[name+'_diff_%d'%d] = df_train[name+'_diff'].shift(d)
        df_test[name+'_diff_%d'%d] = df_test[name+'_diff'].shift(d)
        cpu_feats.append(name+'_diff_%d'%d)

# 按队列名和时间聚合统计
for name in ['CPU_USAGE', 'MEM_USAGE', 'CU_CPU']:
    tdf = df_train.groupby(['TIME_HOUR', 'QUEUE_ID'])[name].agg(
        {'mean', 'median', 'std', 'skew', 'max', 'min'}).reset_index()
    tdf.rename(columns={
        'mean': name+'_T_QID_mean',
        'median': name+'_T_QID_median',
        'std': name+'_T_QID_std',
        'skew': name+'_T_QID_skew',
        'max': name+'_T_QID_max',
        'min': name+'_T_QID_min',
    }, inplace=True)
    cpu_feats = cpu_feats + [name + x for x in [
        '_T_QID_mean', '_T_QID_median', '_T_QID_std', '_T_QID_skew',
        '_T_QID_max', '_T_QID_min']]
    df_train = pd.merge(df_train, tdf, on=['TIME_HOUR', 'QUEUE_ID'], how='left')
    df_test = pd.merge(df_test, tdf, on=['TIME_HOUR', 'QUEUE_ID'], how='left')

df_train = df_train.groupby('QUEUE_ID').apply(make_label).reset_index(drop=True)

print(df_train.shape)


# 斜率
def lr(x1,x2,x3,x4,x5):
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([x1,x2,x3,x4,x5], dtype='float')
    return linregress(x, y)[0]


# 计算给出的5个点CPU_USAGE斜率
df_train['k'] = list(map(lambda x1,x2,x3,x4,x5: lr(x1,x2,x3,x4,x5),
                         df_train.CPU_USAGE_4_ago, df_train.CPU_USAGE_3_ago, df_train.CPU_USAGE_2_ago, df_train.CPU_USAGE_1_ago,
                         df_train.CPU_USAGE))

df_test['k'] = list(map(lambda x1,x2,x3,x4,x5: lr(x1,x2,x3,x4,x5),
                        df_test.CPU_USAGE_4_ago, df_test.CPU_USAGE_3_ago, df_test.CPU_USAGE_2_ago, df_test.CPU_USAGE_1_ago,
                        df_test.CPU_USAGE))
cpu_feats.append('k')

# 类别处理
for name in ['STATUS', 'QUEUE_TYPE', 'PLATFORM', 'RESOURCE_TYPE', 'QUEUE_ID']:
    le = LabelEncoder()
    df_train[name] = le.fit_transform(df_train[name])
    df_test[name] = le.transform(df_test[name])
    df_train[name] = df_train[name].astype('category')
    df_test[name] = df_test[name].astype('category')


print(df_train.shape)
print(df_test.shape)

targets_names = ['CPU_USAGE_1', 'LAUNCHING_JOB_NUMS_1',
                 'CPU_USAGE_2', 'LAUNCHING_JOB_NUMS_2',
                 'CPU_USAGE_3', 'LAUNCHING_JOB_NUMS_3',
                 'CPU_USAGE_4', 'LAUNCHING_JOB_NUMS_4',
                 'CPU_USAGE_5', 'LAUNCHING_JOB_NUMS_5']

df = pd.DataFrame()
df_test = df_test.drop_duplicates(subset=['ID'], keep='last')
df['ID'] = df_test['ID']
print(df.shape)

# 直接利用规则给出job的预测
for name in targets_names:
    df[name] = df_test['LAUNCHING_JOB_NUMS']
score_all = []

for (i, name) in enumerate(targets_names):
    print('===================================================', name)
    if name.split('_')[0] == 'CPU':
        feats = cpu_feats.copy() + targets_names[:i:2]
    elif name.split('_')[0] == 'LAUNCHING':
        df_test[name] = df[name]
        continue
    else:
        continue
    print(feats)

    y = 0
    mse_score = []
    kfold = KFold(n_splits=4, shuffle=True, random_state=2222)
    score = []
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[name])):
        print('--------------------------------------------------Fold ', fold_id)
        train = df_train.loc[trn_idx]
        train_x = train[feats]
        train_y = train[name]
        val = df_train.loc[val_idx]
        val_x = val[feats]
        val_y = val[name]
        print(train_x.shape)

        train_matrix = lgb.Dataset(train_x, label=train_y)
        val_matrix = lgb.Dataset(val_x, label=val_y)
        params = {
            'boosting_type': 'gbdt',
            'num_leaves': 20, 
            'objective': 'mse',
            'learning_rate': 0.05,
            'seed': 2,
            'verbose': -1,
            'nthread': -1,
        }
        model = lgb.train(params, train_matrix, num_boost_round=10000,
                          valid_sets=[train_matrix, val_matrix], verbose_eval=10000, early_stopping_rounds=50)
        y += model.predict(df_test[feats])
        pred_val = model.predict(val_x)
        mse_score.append(mse(pred_val, val_y))
        score.append(get_score((pred_val/10)**2, (val_y/10)**2, name.split('_')[0]))

    print('mse_score: ', np.mean(mse_score))
    print('score: ', np.mean(score))
    score_all.append(np.mean(score))
    df[name] = y/4
    df.loc[df[name] < 0, name] = 0
    if name.split('_')[0] == 'CPU':
        df.loc[df[name] > 100, name] = 100
    df_test[name] = df[name]

# 之前开根号，需要还原
f = ['CPU_USAGE_1', 'CPU_USAGE_2', 'CPU_USAGE_3', 'CPU_USAGE_4', 'CPU_USAGE_5']
for ff in f:
    df[ff] = (df[ff] / 10) ** 2

print(score_all)
print(1-np.sum(score_all))

submit = pd.read_csv('data/submit_example.csv')[['ID']]
submit = pd.merge(submit, np.round(df).astype(int), on='ID')

# CPU后处理 整体扩大
targets_names = ['CPU_USAGE_1', 'CPU_USAGE_2', 'CPU_USAGE_3', 'CPU_USAGE_4', 'CPU_USAGE_5']
for i, name in enumerate(targets_names):
    idx1 = submit[name] >= 80
    idx2 = (submit[name] < 80) & (submit[name] >= 20)

    submit.loc[idx1, name] = submit.loc[idx1, name] * 1.05
    submit.loc[idx2, name] = submit.loc[idx2, name] * (1.1+i*0.01)
    submit.loc[submit[name] > 100, name] = 100

submit = np.round(submit).astype(int)

submit.to_csv('lgb_result.csv', index=False)

