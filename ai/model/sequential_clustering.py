from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from tslearn.metrics import dtw

import numpy as np
import random

# train dataset 집어넣기
dataset = []

B_forehand = np.load('../variable_dataset/Federer_B_Forehand_dataset.npy')
B_backhand = np.load('../variable_dataset/Federer_B_Backhand_dataset.npy')
B_backslice = np.load('../variable_dataset/Federer_B_BackSlice_dataset.npy')
B_forevolley = np.load('../variable_dataset/Federer_B_ForeVolley_dataset.npy')
B_backvolley = np.load('../variable_dataset/Federer_B_BackVolley_dataset.npy')
B_smash = np.load('../variable_dataset/Federer_B_Smash_dataset.npy')
B_serve = np.load('../variable_dataset/Federer_B_Serve_dataset.npy')

for i in B_forehand:
    dataset.append(i)
for i in B_backhand:
    dataset.append(i)
for i in B_backslice:
    dataset.append(i)
for i in B_forevolley:
    dataset.append(i)
for i in B_backvolley:
    dataset.append(i)
for i in B_smash:
    dataset.append(i)
for i in B_serve:
    dataset.append(i)

#random.shuffle(dataset)
dataset = np.array(dataset)
print(dataset.shape)

# 클러스터 수를 2에서 10까지 시도하여 최적의 클러스터 수 찾기
sil_scores = []
db_scores = []

for k in range(2, 20):
    model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=1234)
    labels = model.fit_predict(dataset)

    # 평가 지표 계산 (Silhouette Index와 Davies-Bouldin Index)
    sil_score = silhouette_score(dataset.reshape(dataset.shape[0], -1), labels, metric='euclidean')
    db_score = davies_bouldin_score(dataset.reshape(dataset.shape[0], -1), labels)

    sil_scores.append(sil_score)
    db_scores.append(db_score)

    print(k)

# 결과 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, 20), sil_scores, marker='o')
plt.title('Silhouette Index')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')

plt.subplot(1, 2, 2)
plt.plot(range(2, 20), db_scores, marker='o')
plt.title('Davies-Bouldin Index')
plt.xlabel('Number of clusters')
plt.ylabel('DB Score')

plt.tight_layout()
plt.show()