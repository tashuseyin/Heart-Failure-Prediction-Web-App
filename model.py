# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 02:59:54 2021

@author: Monster
"""

# Giriş

"""
Kardiyovasküler hastalıklar (KVH), her yıl tahminen 17,9 milyon can alarak, dünya çapındaki tüm ölümlerin %31'ini oluşturan, 
küresel olarak 1 numaralı ölüm nedenidir. 5CVD ölümlerinden dördü kalp krizi ve felç nedeniyledir ve bu ölümlerin üçte biri 
70 yaşın altındaki kişilerde erken meydana gelir. Kalp yetmezliği, CVD'lerin neden olduğu yaygın bir olaydır ve bu veri seti, 
olası bir kalp hastalığını tahmin etmek için kullanılabilecek 11 özellik içerir. Kardiyovasküler hastalığı olan veya yüksek kardiyovasküler
risk altında olan kişiler (hipertansiyon, diyabet, hiperlipidemi veya önceden belirlenmiş hastalık gibi bir veya daha fazla risk 
faktörünün varlığı nedeniyle), bir makine öğrenimi modelinin çok yardımcı olabileceği erken tespit ve yönetime ihtiyaç duyar.
"""

# import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, \
    RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# %%

# 1. Load and Check data

data = pd.read_csv("dataset/heart.csv")
print(data.head())
# veri hakkında istatistikler
print(data.describe().T)
print(data.shape)

# %%

# 2. Variable Description
"""
1. Age -> Hastanın yaşı
2. Sex -> Hastanın cinsiyeti
3. ChestPainType -> Göğüs ağrısı tipi [TA: Tipik Angina, ATA: Atipik Angina, NAP: Anjinal olmayan ağrı, ASY: Asemptomatik]
4. RestingBP -> Dinlenme kan basıncı [mm Hg]
5. Cholesterol -> Serum koleströlü [mm/dl]
6. FastingBS -> Açlık kan şekeri [1: if FastingBS > 120 mg/dl, 0: otherwise]
7. RestingECG -> Dinlenme elektrokardiyogram sonuçları [Normal: Normal, ST: ST-T dalga anormalliği olan T dalgası inversiyonları ve/veya ST 
                                                        elevasyonu veya depresyonu > 0.05 mV), LVH: Estes kriterlerine göre olası veya kesin sol ventrikül hipertrofisini gösteriyor]
8. MaxHR -> Ulaşılan maksimum kal atış hızı [60 ile 202 arasında sayısal değer]
9. ExerciseAngina -> egzersize bağlı angina [Y: Evet, N: Hayır]
10. Oldpeak -> oldpeak = ST [Depresyonda ölçülen sayısal değer]
11. ST_Slope -> zirve egzersiz ST segmentinin eğimi [Up: upsloping(eğimli), Flat: flat(düz), Down: dowmsloping(aşağı eğimli)]
12. HeartDisease -> output class [1: heart disease, 0: Normal]
"""

data.info()

# %%

# 3. Univariate Variable Analysis
"""
Categorical variable -> Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope, HeartDisease, FastingBS
Numerical variable -> Age, RestingBP, Cholesterol, MaxHR, Oldpeak
"""


# Categorical Variable
def bar_plot(variable, color):
    var = data[variable]
    varValue = var.value_counts()

    # visualize
    plt.figure(figsize=(9, 3))
    plt.bar(varValue.index, varValue, color=color)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()

    print("{}:\n{}".format(variable, varValue))


category = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease", "FastingBS"]
color_list = ["red", "green", "blue", "orange", "yellow", "purple", "gray"]
for i in range(7):
    bar_plot(category[i], color_list[i])


# Numerical Variable
def kde_plot(variable, color):
    # visualize
    plt.figure(figsize=(9, 6))
    sns.kdeplot(data[variable], color=color, shade=True)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("Kde ile {} dağılımı".format(variable))
    plt.show()


numeric = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
color_list2 = ["red", "green", "blue", "orange", "yellow"]
for i in range(5):
    kde_plot(numeric[i], color_list2[i])

# %%

# 4. Basic Data Analysis

# Sex - HeartDisease
print(data[["Sex", "HeartDisease"]].groupby(["Sex"], as_index=False).mean()
      .sort_values(by="HeartDisease", ascending=False))

# FastingBS - HeartDisease
print(data[["FastingBS", "HeartDisease"]].groupby(["FastingBS"], as_index=False).mean()
      .sort_values(by="HeartDisease", ascending=False))

# ChestPainType - HeartDisease
print(data[["ChestPainType", "HeartDisease"]].groupby(["ChestPainType"], as_index=False).mean()
      .sort_values(by="HeartDisease", ascending=False))

# %%

# 5. Outlier Detection

# plot ile aykırı değer tespiti
list_data = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
for i in list_data:
    sns.boxplot(x=data[i])
    plt.show()


# Aykırı değerleri alt ve üst eşik değerlere baskılama
def find_quantile(df, variable):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    alt_sinir = df[variable].quantile(0.25) - (IQR * 1.5)
    ust_sinir = df[variable].quantile(0.75) + (IQR * 1.5)
    return alt_sinir, ust_sinir


age_alt_sinir, age_ust_sinir = find_quantile(data, "Age")
resting_alt_sinir, resting_ust_sinir = find_quantile(data, "RestingBP")
chol_alt_sinir, chol_ust_sinir = find_quantile(data, "Cholesterol")
max_alt_sinir, max_ust_sinir = find_quantile(data, "MaxHR")
peak_alt_sinir, peak_ust_sinir = find_quantile(data, "Oldpeak")


def count_outlier(df, variable, alt_sinir, ust_sinir):
    x = df[df[variable] < alt_sinir][variable].size
    y = df[df[variable] > ust_sinir][variable].size
    print(variable, "Outlier sayısı: ", x + y)


count_outlier(data, "Age", age_alt_sinir, age_ust_sinir)
count_outlier(data, "RestingBP", resting_alt_sinir, resting_ust_sinir)
count_outlier(data, "Cholesterol", chol_alt_sinir, chol_ust_sinir)
count_outlier(data, "MaxHR", max_alt_sinir, max_ust_sinir)
count_outlier(data, "Oldpeak", peak_alt_sinir, peak_ust_sinir)

# Aykırı değerlere şuanlık bir işlem yapmıyacağım.
# Aykırı değerleri eşik değerlere baskılama yöntemi veya outlier değerleri silebiliriz.

# %%

# 6. Missing Value

print(data.isnull().sum())

# %%

# 7. Visualization

# Correlation Between Age -- RestingBP -- Cholesterol -- MaxHR -- Oldpeak
sns.heatmap(data[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HeartDisease"]].corr(), annot=True)
plt.show()

# tüm datayı görselleme
sns.pairplot(data, hue="HeartDisease")
plt.show()


# %%

def uniqueValue(variable):
    return data[variable].unique()


for i in data.columns:
    print(f"{i}: ", uniqueValue(i))

# %%

# Encoding
# kategorik verileri sayısal hale çevirme
data["Sex"] = data.Sex.map({"M": 0, "F": 1})
data["ChestPainType"] = data.ChestPainType.map({"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3})
data["RestingECG"] = data.RestingECG.map({"Normal": 0, "ST": 1, "LVH": 2})
data["ExerciseAngina"] = data.ExerciseAngina.map({"Y": 1, "N": 0})
data["ST_Slope"] = data.ST_Slope.map({"Up": 1, "Flat": 0, "Down": 2})

data = data.astype(float)
print(data.head())

# %%

# train - test split

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30,
                                                    random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%

# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))

# Stratified K-Fold cross validation

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Hyperparametre tuning with GridSearchCV
knn_params = {"n_neighbors": np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv_model = GridSearchCV(knn, knn_params, cv=skf, scoring="accuracy")
knn_cv_model.fit(X_train, y_train)

print("En iyi skor: {}".format(knn_cv_model.best_score_))
print("En iyi K değeri {}".format(knn_cv_model.best_params_))

# En iyi parametre ile tuned edilmiş modeli kurma
knn = KNeighborsClassifier(n_neighbors=31)
knn_tuned = knn.fit(X_train, y_train)

y_pred = knn_tuned.predict(X_test)
y_pred_train = knn_tuned.predict(X_train)
print("Test Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Train Accuracy Score: ", accuracy_score(y_train, y_pred_train))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

score = cross_val_score(knn_tuned, X_train, y_train, cv=skf, scoring="accuracy")
print("Train score degerleri: ", score)
print("Train  Mean Score:", score.mean())
print("Train Std : " + str(score.std()))

score = cross_val_score(knn_tuned, X_test, y_test, cv=skf, scoring="accuracy")
print("Test score degerleri: ", score)
print("Test Mean Score:", score.mean())
print("Test Std : " + str(score.std()))

pickle.dump(knn_tuned, open("knn_model.pkl", "wb"))

"""
Accuracy Score:  0.8768115942028986
Train score degerleri:  [0.87692308 0.89230769 0.84375    0.875      0.859375   0.890625
 0.84375    0.859375   0.8125     0.8125    ]
Train  Mean Score: 0.856610576923077
Test score degerleri:  [0.85714286 0.89285714 1.         0.85714286 0.78571429 0.92857143
 0.81481481 0.96296296 0.81481481 0.88888889]
Test Mean Score: 0.8802910052910053
En iyi skor: 0.8660576923076924
En iyi K değeri {'n_neighbors': 31}
Test Accuracy Score:  0.8623188405797102
Train Accuracy Score:  0.8644859813084113
[[102  10]
 [ 28 136]]
              precision    recall  f1-score   support

         0.0       0.78      0.91      0.84       112
         1.0       0.93      0.83      0.88       164

    accuracy                           0.86       276
   macro avg       0.86      0.87      0.86       276
weighted avg       0.87      0.86      0.86       276
"""

# %%

# YSA 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import callbacks
from tensorflow.keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.activations import relu, sigmoid


def build_model(layers):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            # Input layer - first hidden layer
            model.add(Dense(16, input_dim=11, activation="relu"))
            model.add(Dropout(0.25))
        else:
            model.add(Dense(nodes))
            model.add(Activation('relu'))
            model.add(Dropout(0.25))
    model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


earlystopping = callbacks.EarlyStopping(monitor='val_loss',
                                        mode='min',
                                        verbose=1,
                                        patience=20)
model = KerasClassifier(build_fn=build_model,
                        callbacks=[earlystopping],
                        validation_data=(X_test, y_test))

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

layers = [[16, 16], [32, 16, 32], [32, 16, 16]]

params = {"layers": layers,
          "batch_size": [32, 64],
          "epochs": [100, 200, 300]}

grid_cv = GridSearchCV(model, param_grid=params, cv=skf, scoring='accuracy')
grid_cv.fit(X_train, y_train,
            validation_data=(X_test, y_test),
            callbacks=[earlystopping])

# %% 

print("En iyi parametreler: {}".format(grid_cv.best_params_))
print("En iyi Skor: {}".format(grid_cv.best_score_))

"""
En iyi parametreler: {'batch_size': 64, 'epochs': 100, 'layers': [32, 16, 16]}
En iyi Skor: 0.867548076923077
"""


# %%

# En iyi parametreler ile tekrardan YSA eğitme
def create_model():
    ann_model = Sequential()
    # Input
    ann_model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
    ann_model.add(Dropout(0.25))

    ann_model.add(Dense(16, activation='relu', kernel_initializer="uniform"))
    ann_model.add(Dropout(0.25))

    ann_model.add(Dense(16, activation='relu', kernel_initializer="uniform"))
    ann_model.add(Dropout(0.25))

    # Dropout
    ann_model.add(Dense(1, activation='sigmoid', kernel_initializer="uniform"))
    optimizer = Adam(learning_rate=0.001)
    ann_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return ann_model


model = create_model()

history = model.fit(X_train, y_train,
                    batch_size=64, epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=[earlystopping])

# %%
model.save("kerasmodel.h5")

# %%

earlystopping = callbacks.EarlyStopping(monitor='val_loss',
                                        mode='min',
                                        verbose=1,
                                        patience=20)
keras_clf = KerasClassifier(create_model,
                            validation_data=(X_test, y_test),
                            epochs=100, batch_size=64,
                            callbacks=[earlystopping])

# Stratified K-Fold cross validation
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

accuracies = cross_val_score(estimator=keras_clf,
                             X=X_train, y=y_train, cv=skf
                             , scoring="accuracy")
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: " + str(mean))
print("Accuracy variancce: " + str(variance))
print(accuracies)

"""
Accuracy mean: 0.8519951923076924
Accuracy variancce: 0.027425145653035957
[0.86153846 0.86153846 0.875      0.84375    0.828125   0.875
 0.875      0.859375   0.78125    0.859375  ]
"""
# %%

# summarize history for acc
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# %%

# load model
from keras.models import load_model

model = load_model("kerasmodel.h5")

# Model Performance

y_pred = model.predict(X_test)
prediction_label = [np.argmax(i) for i in y_pred]
binary_prediction = []

for i in y_pred:
    if i > 0.5:
        binary_prediction.append(1)
    else:
        binary_prediction.append(0)

print(classification_report(y_test, binary_prediction))
print(confusion_matrix(y_test, binary_prediction))
print("Model Score: ", accuracy_score(y_test, binary_prediction))

"""
              precision    recall  f1-score   support

         0.0       0.82      0.90      0.86       112
         1.0       0.93      0.87      0.90       164

    accuracy                           0.88       276
   macro avg       0.87      0.88      0.88       276
weighted avg       0.88      0.88      0.88       276

[[101  11]
 [ 22 142]]
Model Score:  0.8804347826086957
"""
# %%

# load model
from keras.models import load_model

model = load_model("kerasmodel.h5")

# Model Performance

y_pred = model.predict(X_test)
prediction_label = [np.argmax(i) for i in y_pred]
binary_prediction = []

for i in y_pred:
    if i > 0.4:
        binary_prediction.append(1)
    else:
        binary_prediction.append(0)

print(classification_report(y_test, binary_prediction))
print(confusion_matrix(y_test, binary_prediction))
print("Model Score: ", accuracy_score(y_test, binary_prediction))

"""
              precision    recall  f1-score   support

         0.0       0.86      0.90      0.88       112
         1.0       0.93      0.90      0.91       164

    accuracy                           0.90       276
   macro avg       0.89      0.90      0.90       276
weighted avg       0.90      0.90      0.90       276

[[101  11]
 [ 17 147]]
Model Score:  0.8985507246376812
"""
# %%
