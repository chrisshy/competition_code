import numpy as np
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import sys
import sklearn
import joblib
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
from gbdt import MyGradientBoostingClassifier
sys.setrecursionlimit(1000000)
print(1)


# parameter setting
n_estimators= 4000
max_depth= 3
lr=0.1
max_features= 18
min_samples_leaf= 300
criterion ='mse'
multiplier = 10

# model robustness indicator
def calc_vdr(pred, actual, vdr_cutoff = 0.1):
    # Function to calculate VDR at a given cutoff

    df_vdr_train = pd.DataFrame(zip(pred), columns = ['predicted_proba'])
    df_vdr_train['actual'] = np.array(actual)
    df_vdr_train = df_vdr_train.sort_values(by = 'predicted_proba', ascending = False)
    num_bad = df_vdr_train['actual'].sum()
    top_20_pct = int(len(df_vdr_train['actual'])*vdr_cutoff)
    vdr = df_vdr_train.head(top_20_pct)['actual'].sum()/num_bad
    return vdr


# read data
matrix = pd.read_pickle('matrix.pkl')
submission = pd.read_csv('./test_format1.csv')

# split train and test
matrix.fillna(0,inplace=True)
matrix.replace(np.inf,0,inplace=True)
train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['label', 'origin'], axis=1)
del matrix
X =  np.array(train_data.drop(['label'], axis=1))
y = np.array(train_data['label'])
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2,random_state=0)
print(X.shape, y.shape)
# method = SMOTE(random_state =10)
# train_X,train_y = method.fit_resample(train_X,train_y)
print(train_X.shape, train_y.shape)
print(valid_X.shape, valid_y.shape)


# train
model_gbdt = MyGradientBoostingClassifier(n_estimators=n_estimators,max_depth= max_depth,lr=lr ,max_features=max_features,min_samples_leaf=min_samples_leaf,criterion =criterion ,multiplier=multiplier,verbose = True)
model_gbdt.fit(train_X, train_y, valid_X, valid_y)
print('MyGradientBoostingRegressor train_score: {:.4f} valid_score: {:.4f}'.format(
    sklearn.metrics.roc_auc_score(train_y, model_gbdt.predict(train_X)), sklearn.metrics.roc_auc_score(valid_y, model_gbdt.predict(valid_X))))
print(f'train_vdr score: {calc_vdr(model_gbdt.predict(train_X),actual=train_y)}')
print(f'train_vdr score: {calc_vdr(model_gbdt.predict(valid_X),actual=valid_y)}')
str_output = str(n_estimators) +'_'+ str(max_depth).replace('.','') +'_'+ str(lr) +'_'+ str(max_features) +'_'+ str(min_samples_leaf) +'_'+ str(criterion) +'_'+ str(multiplier)
joblib.dump(model_gbdt,'./model_shy/'+ str_output+'_gbdt'+str(sklearn.metrics.roc_auc_score(valid_y, model_gbdt.predict(valid_X))).replace('0.','')+'.model')


# submission output file
prob = model_gbdt.predict(test_data)
submission['prob'] = pd.Series(prob)
submission.to_csv('./submission/BaseLine'+str_output+'_gbdt.csv', index=False)
print(submission.head(5))