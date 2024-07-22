#kütüphaneler
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve,classification_report 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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

target_correlations = app_train.corr()['Target']
target_correlations = target_correlations.drop('Target')  
plt.figure(figsize=(10, 8))
sns.barplot(x=target_correlations.index, y=target_correlations.values)
plt.xticks(rotation=90)
plt.xlabel('Bağımsız Değişkenler')
plt.ylabel('Korelasyon')
plt.title("Bağımsız Değişkenlerin 'Target' ile Korelasyonu")
plt.show()

X = app_train.drop(['Target'], axis=1)
y = app_train['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)

#LGBM
lgbm_model = LGBMClassifier().fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
accuracy_score(y_test, y_pred)

#RandomForest
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred =rf_model.predict(X_test)
accuracy_score(y_test, y_pred)

#Xgboost
from xgboost import XGBClassifier
xgb_model = XGBClassifier().fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred)

#LGBM Model Tuning
lgbm_model = LGBMClassifier().fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
accuracy_score(y_test, y_pred)

lgbm = LGBMClassifier()
lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
              "n_estimators": [200, 500 ,100],
              "max_depth": [1,2,3,5,8]}
lgbm_cv_model = GridSearchCV(lgbm,lgbm_params, cv=10, n_jobs = -1, verbose =2).fit(X_train, y_train)
lgbm_cv_model.best_params_
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

#CatBoost Model Tuning
catb_model = CatBoostClassifier().fit(X_train, y_train , verbose = False)
y_pred = catb_model.predict(X_test)
accuracy_score(y_test, y_pred)

catb= CatBoostClassifier()
catb_params = {"iterations": [200,500,1000],
              "learning_rate": [0.01, 0.03, 0.1],
              "depth":[4,5,8]}

catb_cv_model = GridSearchCV(catb, catb_params, 
                            cv=10, n_jobs=-1,verbose=2).fit(X_train, y_train)

catb_cv_model.best_params_
catb_tuned = CatBoostClassifier(depth= 8,
                               iterations= 200,
                               learning_rate= 0.01).fit(X_train, y_train , verbose= False)
y_pred = catb_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

feature_imp = pd.Series(catb_tuned.feature_importances_,
                       index = X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri"),
plt.show()