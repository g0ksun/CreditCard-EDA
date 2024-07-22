#Kütüphaneler
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#VERİ
applications = pd.read_excel('Applications.xlsx')
applications.shape
applications.info()

#Veri Hakk.

gender = applications.Gender.value_counts(normalize=True)
gender_val = gender.plot.pie(autopct='%1.1f%%')
labels = ['Kadın', 'Erkek']
plt.legend(labels=labels)
plt.title("Başvurularda Cinsiyet Oranı")
plt.show()

car = applications.Own_car.value_counts(normalize=True)
car_val = car.plot.pie(autopct='%1.1f%%')
labels = ['Yok', 'Var']
plt.title("Kredi Başvurusunda Bulunanların Araba Sahiplik Durumları")
plt.legend(labels=labels)

pro = applications.Own_property.value_counts(normalize=True)
pro_val = pro.plot.pie(autopct='%1.1f%%')
labels = ['Var', 'Yok']
plt.title("Kredi Başvurusunda Bulunanların Ev Sahipliği Durumu")
plt.legend(labels=labels)

"""
kontrol için
def count(series):
    return series.value_counts()[1]

selected_columns = ['Work_phone', 'Phone', 'Email']
result1 = applications[selected_columns].apply(count)

def count(series):
    return series.value_counts()[0]

selected_columns = ['Work_phone', 'Phone', 'Email']
result0 = applications[selected_columns].apply(count)

total = result0 + result1
result0_percent = result0 / total * 100
result1_percent = result1 / total * 100

print(result0_percent, result1_percent)
"""


"""
def count_values_in_column(applications, Family_size):
    value_counts = applications[Family_size].value_counts()
    return value_counts

sonuç = count_values_in_column(applications, 'Family_size')
print(sonuç)

def count_values_in_column(applications, Num_children):
    value_counts = applications[Num_children].value_counts()
    return value_counts

sonuç = count_values_in_column(applications, 'Num_children')
print(sonuç)

"""

fsize = applications.Family_size.value_counts()
fsize_val = fsize.plot.bar()
total = len(applications)

for i, v in enumerate(fsize.values):
    percent = (v / total) * 100
    plt.text(i, v, f'{percent:.1f}%', ha='center', va='bottom')
    
plt.xlabel('Birey Sayısı')
plt.ylabel('Frekans')
plt.title("Başvuranın Aile Birey Sayısı")
plt.show()

fsize = applications.Family_size.value_counts()
labels = fsize.index.astype(str)
fsize_val = fsize.plot.pie(labels= None)
plt.legend(labels= labels, loc="center right", bbox_to_anchor=(0.8, 0, 0.5, 1))
plt.show()

child = applications.Num_children.value_counts()
child_val = child.plot.bar() 
plt.xlabel('Çocuk Sayısı')
plt.ylabel('Frekans')
plt.title("Başvuranların Çocuk Sayısı")
plt.show()


une = applications.Unemployed.value_counts(normalize=True)
une_val = une.plot.pie(autopct='%1.1f%%')
labels = ['İşsiz', 'Çalışmakta']
plt.legend(labels=labels)
plt.title("İşsizlik Oranı")
plt.show()

account_duration = applications['Account_duration']
employment_year = applications['Employment_year']
age_year = applications['Age_year']
data = pd.concat([account_duration, employment_year, age_year], axis=1)
correlation_aea = data.corr()

sns.heatmap(correlation_aea, annot=True, cmap='coolwarm')

plt.title('Korelasyon Heatmap')
plt.show()

income_type = applications['Income_type']
total_income = applications['Total_income']

unique_income_types = income_type.unique()

for inc_type in unique_income_types:
    subset = total_income[income_type == inc_type]
    
    plt.hist(subset, bins=20, alpha=0.5, label=inc_type)

plt.xlabel('Gelir')
plt.ylabel('Frekans')
plt.legend()
plt.show()

education = applications['Education']
total_income = applications['Total_income']

unique_educations = education.unique()


for edu in unique_educations:
    subset = total_income[education == edu]
    
    plt.hist(subset, bins=20, alpha=0.5, label=edu)

plt.xlabel('Gelir')
plt.ylabel('Frekans')
plt.legend()
plt.show()

mar = applications.Marital_status.value_counts(normalize=True)
mar_val = mar.plot.pie(autopct='%1.1f%%')

plt.title("Medeni Hal")
plt.show()


hous = applications['Housing_type']
hous_counts = hous.value_counts()
plt.bar(hous_counts.index, hous_counts.values)
plt.title('Barınma Şekli')
plt.ylabel('Frekans')
plt.xticks(rotation=45)
plt.show()

occ = applications.Occupation.value_counts()
labels = occ.index.astype(str)
occ_val = occ.plot.pie(labels= None)
plt.legend(labels= labels, loc="center right", bbox_to_anchor=(1.1, 0, 0.5, 1))
plt.title("Meslek Dağılımı")
plt.show()


##Target Değişkeni: Başvurusunun kabul edilmesini veya reddedilmesini beklediklerimiz 

target = applications.Target.value_counts(normalize=True)
tar_val = target.plot.pie(autopct='%1.1f%%')
labels = ['Düşük Risk', 'Yüksek Risk']
plt.legend(labels=labels)
plt.title("Başvurularda Hedef Kitle Oranı")
plt.show()

gen_tar = applications.groupby(['Gender','Target']).agg({'ID': 'count'})
gender = gen_tar.index.get_level_values('Gender')
target = gen_tar.index.get_level_values('Target')
count = gen_tar['ID']

labels = ['Erkek-Düşük Risk', 'Erkek-Yüksek Risk', 'Kadın-Düşük Risk', 'Kadın-Yüksek Risk']

fig, ax = plt.subplots()
bars= ax.bar(range(len(count)), count)

total = sum(count)
for bar in bars:
    height = bar.get_height()
    percentage = f'{(height/total)*100:.1f}%'
    ax.text(bar.get_x() + bar.get_width() / 2, height, percentage, ha='center', va='bottom')


ax.set_xticks(range(len(count)))
ax.set_xticklabels(labels, rotation=45)

ax.set_xlabel('Gender-Target')
ax.set_ylabel('Count')
ax.set_title('Gender-Target Bar Chart')

plt.tight_layout()
plt.show()

corrall = applications.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corrall, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()