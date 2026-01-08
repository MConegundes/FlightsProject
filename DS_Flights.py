# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 21:02:56 2025

@author: matty
"""

import os
os.cpu_count()

import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.impute import SimpleImputer
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, make_scorer, f1_score
from catboost import CatBoostClassifier, Pool
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler



file_adress = 'C:/Users/matty/OneDrive/Documentos/Projects/Posgrad/Flights_DataScience/'

df_airlines = pd.read_csv(file_adress + 'airlines.csv')
df_airports = pd.read_csv(file_adress + 'airports.csv')
df_flights = pd.read_csv(file_adress + 'flights.csv')

# %% Exploração


print(df_flights.dtypes)
df_flights.isnull().sum()

df_flights['DELAYED'] = (df_flights['WEATHER_DELAY'].notna().astype(int))

cnt = df_flights['DELAYED'].value_counts().sort_index()
plt.figure()
plt.bar(cnt.index, cnt.values)
plt.xlim(-20, 20)
plt.show()

delay_cols = ['AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
              'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
df_flights[delay_cols] = (df_flights[delay_cols].fillna(0).gt(0).astype(int))

delay_counts = df_flights[delay_cols].sum()
plt.figure(figsize=(10, 5))
delay_counts.plot(kind='bar')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

delay_by_airport = (df_flights.groupby('ORIGIN_AIRPORT')['DELAYED'].sum().sort_values(ascending=False).head(30))
plt.figure(figsize=(12, 6))
delay_by_airport.plot(kind='bar')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

delay_by_airport = (df_flights.groupby('AIRLINE')['DELAYED'].sum().sort_values(ascending=False))
plt.figure(figsize=(12, 6))
delay_by_airport.plot(kind='bar')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
                                         

# %% classificação

# %%% Limpeza

df_flights_clean = df_flights[['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'TAIL_NUMBER',
                               'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DISTANCE', 
                               'DELAYED', 'SCHEDULED_ARRIVAL', 'DIVERTED', 'CANCELLED']]
df_flights_clean['ORIGIN_AIRPORT'] = 'o_' + df_flights_clean['ORIGIN_AIRPORT'].str[:]
df_flights_clean['DESTINATION_AIRPORT'] = 'd_' + df_flights_clean['DESTINATION_AIRPORT'].str[:]


df_flights_clean.isnull().sum()

df_flights_clean = df_flights_clean[df_flights_clean['ORIGIN_AIRPORT'].notna()]
df_flights_clean = df_flights_clean[df_flights_clean['TAIL_NUMBER'].notna()]




# %%% catboost

X = df_flights_clean.drop(columns=['DELAYED', 'DIVERTED', 'CANCELLED'])
y = df_flights_clean['DELAYED']

features = ['AIRLINE', 'ORIGIN_AIRPORT', 'TAIL_NUMBER', 'DESTINATION_AIRPORT']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cat_model = CatBoostClassifier(random_state=42, #42
                               cat_features=features, 
                               
                               loss_function='Logloss', #logloss
                               auto_class_weights='Balanced', #Balanced
                               # class_weights=[1, 3],
                               eval_metric='AUC', #AUC
                               
                               iterations=3000, #3000
                               learning_rate=0.05, #0.05
                               
                               depth=9, #7
                               min_data_in_leaf=9, #10
                               
                               reg_lambda=6.5, #5.0
                               bootstrap_type='No', #No #'MVS'; 'No', 'Bernoulli'
                               # subsample=0.8
                               
                               od_type='Iter', #Iter
                               od_wait=200, #200
                               verbose=150 #150
                               )
cat_model.fit(x_train, y_train, 
              eval_set=(x_test, y_test),
              verbose=150)
cat_model.save_model(file_adress + 'DelayedFlights_model.cbm')

# model_loaded = CatBoostClassifier()
# model_loaded.load_model(model_path)

# %%%% feature importance

train_pool = Pool(data= x_train,label= y_train,cat_features= features)
feature_importance = cat_model.get_feature_importance(train_pool)

importance_df = pd.DataFrame({'feature': X.columns,
                              'importance': feature_importance}).sort_values(by='importance', 
                                                                             ascending=False)
                                                                             
plt.figure(figsize=(15, 10))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.title('Feature Importance - CatBoost')
plt.gca().invert_yaxis()
plt.show()                                                                            


# %%% grid search
precision_1 = make_scorer(precision_score, pos_label=1, zero_division=0)
recall_1 = make_scorer(recall_score, pos_label=1)

param_grid = {
    'loss_function': ['Logloss'],
    'auto_class_weights': ['Balanced'],
    'iterations': [3000, 6000],
    'learning_rate': [0.03, 0.05],
    'depth': [6, 9],
    'min_data_in_leaf': [9, 12],
    'reg_lambda': [5.0, 6.5],
    'bootstrap_type': ['No', 'Bernoulli']
    # 'max_features': ['auto', 'sqrt', 'log2'],
    # 'bootstrap_type': ['Bayesian'], #  'MVS'; 'No', 'Bernoulli'
    # 'bagging_temperature': [0.5, 1, 2],
    }

cat_model = CatBoostClassifier(random_state=42,
                               cat_features=features,
                               eval_metric='F1', #F1 AUC
                               od_type='Iter',
                               od_wait=200,
                               verbose=150)

grid_search = GridSearchCV(cat_model, 
                           param_grid, 
                           cv=2, 
                           scoring={'accuracy': 'accuracy', 
                                    'f1': 'f1',
                                    'AUC': 'roc_auc'},
                           refit='f1',
                           # n_jobs=4,
                           verbose=2)
                   

grid_search.best_score_
grid_search.best_params_

results = pd.DataFrame(grid_search.cv_results_)
results.to_excel(file_adress + 'result1.xlsx', index=False) 


# %%% resultados

cat_model.fit(x_train, y_train)

pred_train = cat_model.predict(x_train) ## The predicted values for the train dataset
pred_val = cat_model.predict(x_test) ## The predicted values for the test dataset
accuracy_train = accuracy_score(pred_train, y_train)
accuracy_val = accuracy_score(pred_val, y_test)
recall_val = recall_score(y_test, pred_val, pos_label=1)
precision_val = precision_score(y_test, pred_val, pos_label=1)
f1 = f1_score(y_test, pred_val)


# %% clusterização

# %%% preparação 

df_flights_cluster = df_flights.drop(columns=['TAIL_NUMBER', #'ORIGIN_AIRPORT', 'AIRLINE', 
                                              'DESTINATION_AIRPORT', 'CANCELLATION_REASON'])
df_flights_cluster['DELAY_TOTAL_TIME'] = (df_flights_cluster['AIR_SYSTEM_DELAY'] +
                                          df_flights_cluster['SECURITY_DELAY'] +
                                          df_flights_cluster['AIRLINE_DELAY'] +
                                          df_flights_cluster['LATE_AIRCRAFT_DELAY'] +
                                          df_flights_cluster['WEATHER_DELAY'])
df_flights_cluster['SPEED'] = (df_flights_cluster['DISTANCE'] /
                               df_flights_cluster['AIR_TIME'])
df_flights_cluster['MACH'] = ((df_flights_cluster['SPEED'] * 60) / 767)

airport_delay = (df_flights_cluster
                 .groupby("ORIGIN_AIRPORT")["DELAY_TOTAL_TIME"]
                 .agg(media="mean", mediana="median", maximo="max", 
                      minimo="min", total="sum", quantidade="count").reset_index())

airline_delay = (df_flights_cluster
                 .groupby("AIRLINE")["DELAY_TOTAL_TIME"]
                 .agg(media="mean", mediana="median", maximo="max", 
                      minimo="min", total="sum", quantidade="count").reset_index())

df_flights_cluster = df_flights_cluster.drop(columns=['ORIGIN_AIRPORT', 'AIRLINE'])

df_flights_longos_cluster = df_flights_cluster[df_flights_cluster["DISTANCE"] >= 3000]

# %%% exploração 

correlacao = df_flights_cluster.corr(method="spearman") #spearman / 
plt.figure(figsize=(20, 10))
sns.heatmap(
    correlacao,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Heatmap de Correlação de spearman")
plt.show()

# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['DEPARTURE_TIME'],
#     df_flights_cluster['AIR_TIME']
# )
# plt.xlabel("DEPARTURE_TIME")
# plt.ylabel("AIR_TIME")
# plt.title("Relação entre horário de partida e tempo de voo")
# plt.show()

# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['DEPARTURE_TIME'],
#     df_flights_cluster['DISTANCE']
# )
# plt.xlabel("DEPARTURE_TIME")
# plt.ylabel("DISTANCE")
# plt.title("Relação entre horário de partida e a distancia")
# plt.show()


# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['AIR_TIME'],
#     df_flights_cluster['DISTANCE']
# )
# plt.xlabel("AIR_TIME")
# plt.ylabel("DISTANCE")
# plt.title("Relação entre tempo de voo e a distancia")
# plt.show()

plt.figure(figsize=(20, 10))
plt.scatter(
    df_flights_cluster['MACH'],
    df_flights_cluster['AIR_TIME']
)
plt.xlabel("MACH")
plt.ylabel("AIR_TIME")
plt.show()

# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['SPEED'],
#     df_flights_cluster['AIR_TIME']
# )
# plt.xlabel("SPEED")
# plt.ylabel("AIR_TIME")
# plt.show()

# fig = plt.figure(figsize=(20, 60))
# ax = fig.add_subplot(111, projection="3d")
# scatter = ax.scatter(
#     df_flights_cluster['AIR_TIME'],
#     df_flights_cluster['DISTANCE'],
#     df_flights_cluster['MACH'],
#     s=60)
# ax.set_xlabel("AIR_TIME")
# ax.set_ylabel("DISTANCE")
# ax.set_zlabel("MACH")
# fig.colorbar(scatter, label="Mach")
# plt.show()

plt.figure(figsize=(20, 10))
plt.scatter(
    df_flights_longos_cluster['DEPARTURE_TIME'],
    df_flights_longos_cluster['DISTANCE']
)
plt.xlabel("DEPARTURE_TIME")
plt.ylabel("DISTANCE")
plt.title("Relação entre horário de partida e a distancia")
plt.show()

plt.figure(figsize=(20, 10))
plt.scatter(
    df_flights_longos_cluster['ARRIVAL_TIME'],
    df_flights_longos_cluster['DISTANCE']
)
plt.xlabel("ARRIVAL_TIME")
plt.ylabel("DITANCE")
plt.title("Relação entre distancia e horario de chegada")
plt.show()
             
df_flights_longos_cluster.isnull().sum()
len(df_flights_longos_cluster)

# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['DAY_OF_WEEK'],
#     df_flights_cluster['DEPARTURE_TIME']
# )
# plt.xlabel("DAY_OF_WEEK")
# plt.ylabel("DEPARTURE_TIME")
# plt.show()


# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['DAY_OF_WEEK'],
#     df_flights_cluster['ARRIVAL_TIME']
# )
# plt.xlabel("DAY_OF_WEEK")
# plt.ylabel("ARRIVAL_TIME")
# plt.show()


# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['DAY_OF_WEEK'],
#     df_flights_cluster['DISTANCE']
# )
# plt.xlabel("DAY_OF_WEEK")
# plt.ylabel("DISTANCE")
# plt.show()

# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['MONTH'],
#     df_flights_cluster['DISTANCE']
# )
# plt.xlabel("MONTH")
# plt.ylabel("DISTANCE")
# plt.show()

# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['DAY_OF_WEEK'],
#     df_flights_cluster['AIR_TIME']
# )
# plt.xlabel("DAY_OF_WEEK")
# plt.ylabel("AIR_TIME")
# plt.show()

# plt.figure(figsize=(20, 10))
# plt.scatter(
#     df_flights_cluster['MONTH'],
#     df_flights_cluster['AIR_TIME']
# )
# plt.xlabel("MONTH")
# plt.ylabel("AIR_TIME")
# plt.show()

dt_kmeans = df_flights_longos_cluster[['DEPARTURE_TIME','DISTANCE']]
dt_kmeans = dt_kmeans[dt_kmeans['DEPARTURE_TIME'].notna()]
dt_kmeans.isnull().sum()
len(dt_kmeans)

ss = StandardScaler()
ss.fit(dt_kmeans)
normalized_data = ss.transform(dt_kmeans)

# %%% Kmeans

cluster = KMeans(n_clusters=4, random_state=42)
cluster.fit(dt_kmeans)

centroids = cluster.cluster_centers_
labels = cluster.labels_

colors = ["g.","r.","b.","y.","m."]
dataset_array = np.array(dt_kmeans)
for i in range(len(dataset_array)):
    plt.plot(dataset_array[i][0], dataset_array[i][1], colors [labels[i]], markersize =25);
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150);
plt.show()


K = range(1,10)
Sum_of_squared_distances = []
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(normalized_data)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('No de Clusters')
plt.ylabel('Soma_das_distancias_quadradas')
plt.title('Metodo do Cotovelo para o k Ótimo')
plt.show()
