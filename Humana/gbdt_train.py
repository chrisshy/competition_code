import numpy as np
import pandas as pd
import sys
import sklearn
import joblib
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE


from gbdt import MyGradientBoostingClassifier
sys.setrecursionlimit(1000000)
print(1)

def calc_vdr(pred, actual, vdr_cutoff = 0.1):
    # Function to calculate VDR at a given cutoff

    df_vdr_train = pd.DataFrame(zip(pred), columns = ['predicted_proba'])
    df_vdr_train['actual'] = np.array(actual)
    df_vdr_train = df_vdr_train.sort_values(by = 'predicted_proba', ascending = False)
    num_bad = df_vdr_train['actual'].sum()
    top_20_pct = int(len(df_vdr_train['actual'])*vdr_cutoff)
    vdr = df_vdr_train.head(top_20_pct)['actual'].sum()/num_bad
    return vdr

data_cleaned = pd.read_pickle('D:/Humana/dataset_test/Process/data_mutual_info_final.pkl')
print(data_cleaned)
X =  np.array(data_cleaned.iloc[:,:-1])
y = np.array(data_cleaned.iloc[:,-1])

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2,random_state=0)
# print(X.shape, y.shape)
# method = SMOTE(random_state =10)
# train_X,train_y = method.fit_resample(train_X,train_y)
print(train_X.shape, train_y.shape)
print(valid_X.shape, valid_y.shape)

model_gbdt = MyGradientBoostingClassifier(n_estimators=500,max_depth= 5,lr=0.3 ,max_features='sqrt',min_samples_leaf=100,criterion ='mae' ,verbose = True)
model_gbdt.fit(train_X, train_y, valid_X, valid_y)
print('MyGradientBoostingRegressor train_score: {:.4f} valid_score: {:.4f}'.format(
    sklearn.metrics.roc_auc_score(train_y, model_gbdt.predict(train_X)), sklearn.metrics.roc_auc_score(valid_y, model_gbdt.predict(valid_X))))

print(f'train_vdr score: {calc_vdr(model_gbdt.predict(train_X),actual=train_y)}')
print(f'train_vdr score: {calc_vdr(model_gbdt.predict(valid_X),actual=valid_y)}')

joblib.dump(model_gbdt,'./model_shy/500_3_0.2_sqrt_500_mse_gbdt.model')
