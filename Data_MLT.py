#Kütüphaneler
import numpy as np 
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


applications1 = pd.read_excel('Applications.xlsx')
applications =applications1.copy()
applications.shape
applications.info()


app_train = applications.drop(['ID'], axis=1)
 
def creatingDummyVariables(df, columns) :
    dummy1 = pd.get_dummies(df[columns], drop_first=True)
    df1 = pd.concat([df, dummy1], axis=1)
    df1.drop(columns, axis = 1, inplace = True)
    
    return df1

app_train = creatingDummyVariables(app_train, ['Income_type', 'Education', 'Marital_status', 'Housing_type', 'Occupation'])
#özellik seçimi
target_correlations = app_train.corr()['Target']
target_correlations = target_correlations.drop('Target')  
plt.figure(figsize=(10, 8))
sns.barplot(x=target_correlations.index, y=target_correlations.values)
plt.xticks(rotation=90)
plt.xlabel('Bağımsız Değişkenler')
plt.ylabel('Korelasyon')
plt.title("Bağımsız Değişkenlerin 'Target' ile Korelasyonu")
plt.show()

features = app_train.iloc[:, np.r_[0:13, 14:49]].columns.tolist()
target = 'Target'

p_values = {}
for feature in features:
    p_value = stats.ttest_ind(app_train[app_train[target] == 0][feature], app_train[app_train[target] == 1][feature]).pvalue
    p_values[feature] = p_value
    
    
sorted_p_values = sorted(p_values.items(), key=lambda x: x[1])
pv_df = pd.DataFrame(sorted_p_values, columns=['Feature', 'P-value'])

def normal(sayi):
    return "{:.4f}".format(sayi)

pv_df['P-value'] = pv_df['P-value'].apply(normal)
pv_df.head()
'''
#Model Building for LR

def logisticReg (df) :

    X = df.drop(['Target'], axis=1)
    y = df['Target']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)
    
    logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
    res = logm1.fit()
    return res

res = logisticReg(app_train)
res.summary()
'''

X = app_train.drop(['Target'], axis=1)
y = app_train['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)


X_balance,Y_balance = SMOTE().fit_resample(X_train,y_train)
X_balance = pd.DataFrame(X_balance, columns = X_train.columns)
Y_balance = pd.DataFrame(Y_balance, columns=["Target"])

def calc_iv(df, feature, target, pr=False):
    
    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())

    iv = data['IV'].sum()

    return iv, data

iv_df = X_balance.copy()
iv_df["target"] = y_train

features = iv_df.columns[:-1].tolist()

iv_list = []
for feature in features:
    iv, data = calc_iv(iv_df, feature, 'target')
    iv_list.append(round(iv,4))

woe_df = pd.DataFrame(np.column_stack([features, iv_list]), 
                      columns=['Feature', 'iv'])
woe_df

scaler = StandardScaler()
scaler.fit(X_balance)

X_train = pd.DataFrame(scaler.transform(X_balance), columns=[X_balance.columns])

model = LogisticRegression(solver='liblinear')
rfe = RFE(model)
fit = rfe.fit(X_train, Y_balance)
rfe_features = pd.DataFrame({"Feature":features,
              "Support_LogisticRegression":fit.support_,
              "Feature_Rank_logisticRegression":fit.ranking_})
rfe_features

model = ExtraTreesClassifier(n_estimators=10)
model.fit(X_balance, Y_balance)
feature_importances = pd.DataFrame({"Feature":features,
              "Feature_Importance_ExtratreeClassifier":model.feature_importances_})

df1=pd.merge(woe_df, feature_importances, on=["Feature"])
df2 = pd.merge(df1, rfe_features, on=["Feature"])
feature_selection_df = pd.merge(df2, pv_df, on=["Feature"])
feature_selection_df['P-value'] = feature_selection_df['P-value'].astype(float)
feature_selection_df.sort_values(by="iv",ascending=False)
#### 
filtered_df  = feature_selection_df[(feature_selection_df['Support_LogisticRegression'] == True) | (feature_selection_df['P-value'] < 0.05)]
filtered_df.sort_values(by="iv",ascending=False)


###

classifiers = {
    "LogisticRegression" : LogisticRegression(),
    "KNeighbors" : KNeighborsClassifier(),
    "DecisionTree" : DecisionTreeClassifier(),
    "RandomForest" : RandomForestClassifier(n_estimators=250,max_depth=12,min_samples_leaf=16),
    "XGBoost" : XGBClassifier(max_depth=12,
                              n_estimators=250,
                              min_child_weight=8, 
                              subsample=0.8, 
                              learning_rate =0.02,    
                              seed=42),
    "CatBoost" : CatBoostClassifier(iterations=250,
                           learning_rate=0.2,
                           od_type='Iter',
                           verbose=25,
                           depth=16,
                           random_seed=42)
}

result_table = pd.DataFrame(columns=['classifiers','accuracy','presicion','recall','f1_score','fpr','tpr','auc'])

y_test = y_test.astype(int)


for key, classifier in classifiers.items():
    classifier.fit(X_balance[filtered_df], Y_balance)
    y_predict = classifier.predict(X_test[filtered_df])
    
    yproba = classifier.predict_proba(X_test[filtered_df])[::,1]
    
   
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    conf_matrix = confusion_matrix(y_test,y_predict)
    
    result_table = result_table.append({'classifiers':key,
                                        'accuracy':accuracy_score(y_test, y_predict),
                                        'presicion':precision_score(y_test, y_predict, average='weighted'),
                                        'recall':recall_score(y_test, y_predict, average='weighted'),
                                        'f1_score':f1_score(y_test, y_predict, average='weighted'),
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc
                                         }, ignore_index=True)
        
result_table.set_index('classifiers', inplace=True)


###model tuning

lgbm = LGBMClassifier()
lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
              "n_estimators": [200, 500 ,100],
              "max_depth": [1,2,3,5,8]}
lgbm_cv_model = GridSearchCV(lgbm,lgbm_params, cv=10, n_jobs = -1, verbose =2).fit(X_train, y_train)

lgbm_cv_model.bestparams
lgbm_tuned = LGBMClassifier(learning_rate= 0.1, 
                            max_depth= 5, 
                            n_estimators= 500).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                       index = X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri"),
plt.show()