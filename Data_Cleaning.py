#Kütüphaneler
import numpy as np 
import pandas as pd 

#Veri Setleri
df_app = pd.read_csv('application_record.csv')
df_cre = pd.read_csv('credit_record.csv')

df_app.head()
df_cre.head()

#data cleaning for application_record data
df_app.dtypes
df_app[["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"]].head() #float görünen sütunlar incelendi
df_app = df_app.astype({"CNT_FAM_MEMBERS":'int', "AMT_INCOME_TOTAL":'int'}) 
df_app.dtypes

#object görünen sütunlar incelendi
df_app[["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]].head()
#değişken atanarak int yapıldı
df_app['CODE_GENDER'].replace('M',0,inplace=True)
df_app['CODE_GENDER'].replace('F',1,inplace=True)
df_app['FLAG_OWN_CAR'].replace('Y',1,inplace=True)
df_app['FLAG_OWN_CAR'].replace('N',0,inplace=True)
df_app['FLAG_OWN_REALTY'].replace('Y',1,inplace=True)
df_app['FLAG_OWN_REALTY'].replace('N',0,inplace=True)

#kategorik olanlar belirtildi
df_app['NAME_INCOME_TYPE'] = df_app['NAME_INCOME_TYPE'].astype('category')
df_app['NAME_EDUCATION_TYPE'] = df_app['NAME_EDUCATION_TYPE'].astype('category')
df_app['NAME_FAMILY_STATUS'] = df_app['NAME_FAMILY_STATUS'].astype('category')
df_app['NAME_HOUSING_TYPE'] = df_app['NAME_HOUSING_TYPE'].astype('category')
df_app['OCCUPATION_TYPE'] = df_app['OCCUPATION_TYPE'].astype('category')

df_app.dtypes
df_app.head(10)
df_app.info() ## boş sütunlar var
df_app['OCCUPATION_TYPE'] = df_app['OCCUPATION_TYPE'].cat.add_categories(['Other'])
df_app['OCCUPATION_TYPE'].fillna(value='Other', inplace=True)#NA değerleri other ile doldurdum
df_app.isnull().sum()

#sütunları yeniden isimlendirdim
df_app = df_app.rename(columns={'CODE_GENDER': 'Gender', 'FLAG_OWN_CAR': 'Own_car', 'FLAG_OWN_REALTY': 'Own_property', 'CNT_CHILDREN': 'Num_children', 'AMT_INCOME_TOTAL': 'Total_income', 'NAME_INCOME_TYPE': 'Income_type', 'NAME_EDUCATION_TYPE': 'Education', 'NAME_FAMILY_STATUS': 'Marital_status', 'NAME_HOUSING_TYPE': 'Housing_type', 'DAYS_BIRTH': 'Birthday', 'DAYS_EMPLOYED': 'Employment_duration', 'FLAG_MOBIL': 'Mobil_phone', 'FLAG_WORK_PHONE': 'Work_phone', 'FLAG_PHONE': 'Phone', 'FLAG_EMAIL': 'Email', 'OCCUPATION_TYPE': 'Occupation', 'CNT_FAM_MEMBERS': 'Family_size'})
df_app.nunique() #sütunlardaki benzersiz değerleri inceledim


#sürekli değişken oluşturma
df_app['Age_year'] = (-df_app['Birthday'] / 365.2425).round().astype(int)

df_app['Unemployed']=0
df_app.loc[-df_app['Employment_duration']<0,'Unemployed']=1

df_app['Employment_year'] = (-df_app['Employment_duration'] / 365.2425).round().astype(int)
df_app.loc[df_app['Employment_year'] < 0, 'Employment_year'] = 0


#data cleaning for credit_record data
df_cre.isnull().sum()
df_cre.head(8)
df_cre['STATUS'] = df_cre['STATUS'].replace({'C': 0, 'X': 0})
df_cre.dtypes
df_cre['STATUS'] = df_cre['STATUS'].astype(int)
df_cre.dtypes
df_cre.nunique()# 45985 kişinin kredi kaydı varken başvuran 438510 kişi vardı

#Yeni değişkenler oluşturma
df_cre['Target']=df_cre['STATUS']
df_cre.loc[df_cre['Target']>=1,'Target']=1

df_target=pd.DataFrame(df_cre.groupby(['ID'])['Target'].agg(max)).reset_index()


df_start=pd.DataFrame(df_cre.groupby(['ID'])['MONTHS_BALANCE'].agg(min)).reset_index()
df_start.rename(columns={'MONTHS_BALANCE':'Account_duration'}, inplace=True)
df_start['Account_duration']=-df_start['Account_duration']



#Veri setinde sütunları düzenleme
a=pd.merge(df_app, df_target, how='inner', on=['ID'])
b=pd.merge(a, df_start, how='inner', on=['ID'])
new_app= b

new_app.columns
new_app = df_app.drop(['Mobil_phone','Birthday', 'Employment_duration'], axis=1, inplace=True)


new_app=new_app[['ID', 'Gender', 'Own_car', 'Own_property', 'Work_phone',
               'Phone', 'Email', 'Unemployed', 'Num_children', 'Family_size','Age_year', 
               'Account_duration', 'Employment_year', 'Total_income',    
               'Income_type', 'Education', 'Marital_status',
               'Housing_type', 'Occupation','Target']]

new_app.drop(new_app[(new_app['Family_size'] == 1) & (new_app['Num_children'] > 0)].index, inplace=True)

new_app.to_excel("C:\\Users\\hp\\Desktop\\Bitirme Projem\\Applications.xlsx", index=False)

