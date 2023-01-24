import warnings

from sklearn import metrics, __all__

warnings.filterwarnings('ignore')
import inline
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.datasets import make_classification, make_blobs


df = pd.read_csv('food-allergy-analysis-Zenodo.csv')

df.drop("SUBJECT_ID", axis=1, inplace=True)

# LABEL ENCODING
df_2 = df.copy()

df_2["GENDER_FACTOR"] = df_2["GENDER_FACTOR"].astype("category")
df_2["GENDER_FACTOR"] = df_2["GENDER_FACTOR"].cat.codes
df_2["RACE_FACTOR"] = df_2["RACE_FACTOR"].astype("category")
df_2["RACE_FACTOR"] = df_2["RACE_FACTOR"].cat.codes
df_2["ETHNICITY_FACTOR"] = df_2["ETHNICITY_FACTOR"].astype("category")
df_2["ETHNICITY_FACTOR"] = df_2["ETHNICITY_FACTOR"].cat.codes
df_2["PAYER_FACTOR"] = df_2["PAYER_FACTOR"].astype("category")
df_2["PAYER_FACTOR"] = df_2["PAYER_FACTOR"].cat.codes
df_2["ATOPIC_MARCH_COHORT"] = df_2["ATOPIC_MARCH_COHORT"].astype("category")
df_2["ATOPIC_MARCH_COHORT"] = df_2["ATOPIC_MARCH_COHORT"].cat.codes

#Veri setindeki NaN değerleri 0 atıyoruz
df_2[["SHELLFISH_ALG_START", "SHELLFISH_ALG_END", "FISH_ALG_START", "FISH_ALG_END", "MILK_ALG_START", "MILK_ALG_END", "SOY_ALG_START", "SOY_ALG_END",
      "EGG_ALG_START", "EGG_ALG_END", "WHEAT_ALG_START", "WHEAT_ALG_END", "PEANUT_ALG_START", "PEANUT_ALG_END", "SESAME_ALG_START", "SESAME_ALG_END",
      "TREENUT_ALG_START", "TREENUT_ALG_END", "WALNUT_ALG_START", "WALNUT_ALG_END", "PECAN_ALG_START", "PECAN_ALG_END", "PISTACH_ALG_START", "PISTACH_ALG_END",
      "ALMOND_ALG_START", "ALMOND_ALG_END", "BRAZIL_ALG_START", "BRAZIL_ALG_END", "HAZELNUT_ALG_START", "HAZELNUT_ALG_END", "CASHEW_ALG_START", "CASHEW_ALG_END",
      "ATOPIC_DERM_START", "ATOPIC_DERM_END", "ALLERGIC_RHINITIS_START", "ALLERGIC_RHINITIS_END", "ASTHMA_START", "ASTHMA_END", "FIRST_ASTHMARX", "LAST_ASTHMARX",
      "NUM_ASTHMARX"]] = \
    df[["SHELLFISH_ALG_START", "SHELLFISH_ALG_END", "FISH_ALG_START", "FISH_ALG_END", "MILK_ALG_START", "MILK_ALG_END", "SOY_ALG_START", "SOY_ALG_END",
        "EGG_ALG_START", "EGG_ALG_END", "WHEAT_ALG_START", "WHEAT_ALG_END", "PEANUT_ALG_START", "PEANUT_ALG_END", "SESAME_ALG_START", "SESAME_ALG_END",
        "TREENUT_ALG_START", "TREENUT_ALG_END", "WALNUT_ALG_START", "WALNUT_ALG_END", "PECAN_ALG_START", "PECAN_ALG_END", "PISTACH_ALG_START", "PISTACH_ALG_END",
        "ALMOND_ALG_START", "ALMOND_ALG_END", "BRAZIL_ALG_START", "BRAZIL_ALG_END", "HAZELNUT_ALG_START", "HAZELNUT_ALG_END", "CASHEW_ALG_START", "CASHEW_ALG_END",
        "ATOPIC_DERM_START", "ATOPIC_DERM_END", "ALLERGIC_RHINITIS_START", "ALLERGIC_RHINITIS_END", "ASTHMA_START", "ASTHMA_END", "FIRST_ASTHMARX", "LAST_ASTHMARX",
        "NUM_ASTHMARX"]].fillna(0)

#print(df_2)

# boş değerleri kontrol ediyoruz
df.info()

# Tüm verilerin ort gibi değerlerini veriyor
describe = df.describe()
print(describe)

# nul değerinin sayısını belirtiyor
a = df.isnull().sum()
print(a)

b = df.nunique()
print(b)

f = df_2['GENDER_FACTOR'].unique()
f = df_2['RACE_FACTOR'].unique()
f = df_2['ETHNICITY_FACTOR'].unique()
f = df_2['PAYER_FACTOR'].unique()
f = df_2['ATOPIC_MARCH_COHORT'].unique()
f = df_2['SHELLFISH_ALG_START'].unique()
f = df_2['SHELLFISH_ALG_END'].unique()
#print(f)

# aradaki ilişki
d = df_2.corr()
print(d)

# data visualization
sns.heatmap(df_2.corr(), annot=True)
plt.show()

# pasta grafiği
df.BIRTH_YEAR.value_counts().plot(kind = "pie")
plt.show()
df.GENDER_FACTOR.value_counts().plot(kind = "pie")
plt.show()
df.RACE_FACTOR.value_counts().plot(kind = "pie")
plt.show()
df.ETHNICITY_FACTOR.value_counts().plot(kind = "pie")
plt.show()
df.PAYER_FACTOR.value_counts().plot(kind = "pie")
plt.show()
df.ATOPIC_MARCH_COHORT.value_counts().plot(kind = "pie")
plt.show()

# karşılaştırmalı grafik
sns.barplot(x = "GENDER_FACTOR", y = "SHELLFISH_ALG_END", hue="RACE_FACTOR", data=df)
plt.show()
sns.barplot(x = "BIRTH_YEAR", y = "SHELLFISH_ALG_END", hue="ETHNICITY_FACTOR", data=df)
plt.show()

# Gender factorun sayısla karşılıklarını veriyor
count = df.BIRTH_YEAR.value_counts()
print(count)
count = df.GENDER_FACTOR.value_counts()
print(count)
count = df.RACE_FACTOR.value_counts()
print(count)
count = df.ETHNICITY_FACTOR.value_counts()
print(count)
count = df.PAYER_FACTOR.value_counts()
print(count)
count = df.ATOPIC_MARCH_COHORT.value_counts()
print(count)

# Test ve train ayırma
X = df_2.drop('ATOPIC_MARCH_COHORT', axis=1)
Y = df_2['ATOPIC_MARCH_COHORT']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#print(X_train)
#print(X_test)

# K-NN (K-NEAREST NEİGHBORS CLUSTERİNG)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# knn modeli oluşturuldu
knn = KNeighborsClassifier(n_neighbors=5)
#print(knn)

# knn modeli eğitildi
modell = knn.fit(X_train, y_train)
#print(modell)

knn_score = knn.score(X_test, y_test)
print('-KNN Score: ', knn_score)

print('-Confusion Matrix-')
y_predd = knn.predict(X_test)
cm = confusion_matrix(y_test, y_predd)
print(cm)

import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.title('Predicted/Truth')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

knn = KNeighborsRegressor(n_neighbors=5,weights='uniform',algorithm='auto')
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report
print('-Classification Report-')
print(classification_report(y_test, y_pred))
print('-Confusion Matrix-')
print(confusion_matrix(y_test, y_pred))

df00 = df_2[:100000]
df11 = df_2[100000:200000]
df22 = df_2[200000:]

plt.xlabel('BIRTH_YEAR')
plt.ylabel('SHELLFISH_ALG_START')
plt.scatter(df00['BIRTH_YEAR'], df00['SHELLFISH_ALG_START'], color='green', marker='+')
plt.scatter(df11['BIRTH_YEAR'], df11['SHELLFISH_ALG_START'], color='blue')
plt.scatter(df22['BIRTH_YEAR'], df22['SHELLFISH_ALG_START'], color='black', marker='*')
plt.show()

plt.xlabel('BIRTH_YEAR')
plt.ylabel('SHELLFISH_ALG_END')
plt.scatter(df00['BIRTH_YEAR'], df00['SHELLFISH_ALG_END'], color='green', marker='+')
plt.scatter(df11['BIRTH_YEAR'], df11['SHELLFISH_ALG_END'], color='blue')
plt.scatter(df22['BIRTH_YEAR'], df22['SHELLFISH_ALG_END'], color='black', marker='*')
plt.show()

plt.xlabel('AGE_START_YEARS')
plt.ylabel('SHELLFISH_ALG_START')
plt.scatter(df00['AGE_START_YEARS'], df00['SHELLFISH_ALG_START'], color='green', marker='+')
plt.scatter(df11['AGE_START_YEARS'], df11['SHELLFISH_ALG_START'], color='blue')
plt.scatter(df22['AGE_START_YEARS'], df22['SHELLFISH_ALG_START'], color='black', marker='*')
plt.show()

plt.xlabel('AGE_START_YEARS')
plt.ylabel('SHELLFISH_ALG_END')
plt.scatter(df00['AGE_START_YEARS'], df00['SHELLFISH_ALG_END'], color='green', marker='+')
plt.scatter(df11['AGE_START_YEARS'], df11['SHELLFISH_ALG_END'], color='blue')
plt.scatter(df22['AGE_START_YEARS'], df22['SHELLFISH_ALG_END'], color='black', marker='*')
plt.show()

plt.xlabel('AGE_END_YEARS')
plt.ylabel('SHELLFISH_ALG_START')
plt.scatter(df00['AGE_END_YEARS'], df00['SHELLFISH_ALG_START'], color='green', marker='+')
plt.scatter(df11['AGE_END_YEARS'], df11['SHELLFISH_ALG_START'], color='blue')
plt.scatter(df22['AGE_END_YEARS'], df22['SHELLFISH_ALG_START'], color='black', marker='*')
plt.show()

plt.xlabel('AGE_END_YEARS')
plt.ylabel('SHELLFISH_ALG_END')
plt.scatter(df00['AGE_END_YEARS'], df00['SHELLFISH_ALG_END'], color='green', marker='+')
plt.scatter(df11['AGE_END_YEARS'], df11['SHELLFISH_ALG_END'], color='blue')
plt.scatter(df22['AGE_END_YEARS'], df22['SHELLFISH_ALG_END'], color='black', marker='*')
plt.show()

# K MEANS CLUSTERİNG ALGORİTHM
plt.scatter(df_2['BIRTH_YEAR'], df_2['SHELLFISH_ALG_START'])
plt.xlabel('BIRTH_YEAR')
plt.ylabel('SHELLFISH_ALG_START')
plt.show()
plt.scatter(df_2['BIRTH_YEAR'], df_2['SHELLFISH_ALG_END'])
plt.xlabel('BIRTH_YEAR')
plt.ylabel('SHELLFISH_ALG_END')
plt.show()
plt.scatter(df_2['AGE_START_YEARS'], df_2['SHELLFISH_ALG_START'])
plt.xlabel('AGE_START_YEARS')
plt.ylabel('SHELLFISH_ALG_START')
plt.show()
plt.scatter(df_2['AGE_START_YEARS'], df_2['SHELLFISH_ALG_END'])
plt.xlabel('AGE_START_YEARS')
plt.ylabel('SHELLFISH_ALG_END')
plt.show()
plt.scatter(df_2['AGE_END_YEARS'], df_2['SHELLFISH_ALG_START'])
plt.xlabel('AGE_END_YEARS')
plt.ylabel('SHELLFISH_ALG_START')
plt.show()
plt.scatter(df_2['AGE_END_YEARS'], df_2['SHELLFISH_ALG_END'])
plt.xlabel('AGE_END_YEARS')
plt.ylabel('SHELLFISH_ALG_END')
plt.show()

km = KMeans(n_clusters=5)
#print(km)

y_predicted = km.fit_predict(df_2[['AGE_START_YEARS', 'SHELLFISH_ALG_END']])
#print(y_predicted)

df_2['cluster'] = y_predicted
df_2.head()

df1 = df_2[df_2.cluster==0]
df2 = df_2[df_2.cluster==1]
df3 = df_2[df_2.cluster==2]

plt.scatter(df1['AGE_START_YEARS'],df1['SHELLFISH_ALG_END'], color='green')
plt.scatter(df2['AGE_START_YEARS'],df2['SHELLFISH_ALG_END'], color='red')
plt.scatter(df3['AGE_START_YEARS'],df3['SHELLFISH_ALG_END'], color='black')

plt.xlabel('Age start years')
plt.ylabel('Shellfish alg. end')
plt.show()

plt.scatter(df1['BIRTH_YEAR'],df1['SHELLFISH_ALG_END'], color='green')
plt.scatter(df2['BIRTH_YEAR'],df2['SHELLFISH_ALG_END'], color='red')
plt.scatter(df3['BIRTH_YEAR'],df3['SHELLFISH_ALG_END'], color='black')

plt.xlabel('Birth year')
plt.ylabel('Shellfish alg. end')
plt.show()

xx = df_2['AGE_START_YEARS'].unique()
yy = df_2['SHELLFISH_ALG_END'].unique()

data = list(zip(xx,yy))
#print(data)

inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('The Elbow Method')
plt.show()

# LOGİSTİC REGRESSİON

model = LogisticRegression()

model.fit(X_train, y_train)

# Accuracy Score
# accuracy on training data
X_train_prediction = model.predict(X_train)
print(X_train_prediction)

training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Training Data Accuracy Score: ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)

test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Test Data Accuracy Score: ', test_data_accuracy)

z, t = make_classification(
    n_samples=100,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0
)
#print(z)
#print(t)

plt.scatter(z, t, c=t, cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.show()

X_train_shape = X_train.shape
print('-X Train Shape: ', X_train_shape)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print('-Confusion Matrix-')
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# HİERARCHİCAL CLUSTERİNG
from scipy.cluster.hierarchy import dendrogram, linkage

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.title('Dendrogram')
plt.xlabel('AGE_START_YEARS')
plt.ylabel('SHELLFISH_ALG_END')
plt.show()
