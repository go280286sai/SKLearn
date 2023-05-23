"""
====================================================================
AffinityPropagation создает кластеры, отправляя сообщения между парами выборок до сходимости
====================================================================
class sklearn.cluster.AffinityPropagation(*, damping=0.5, max_iter=200, convergence_iter=15,
copy=True, preference=None, affinity='euclidean', verbose=False, random_state=None)
"""
import pandas as pd
from sklearn.cluster import AffinityPropagation

from index import Index


class AffinityPropagationClass(Index):
    def getData(self):
        self.model = AffinityPropagation(random_state=3)
        self.model.fit(self.X)
        self.df['index'] = labels = self.model.labels_
        self.centroids = self.model.cluster_centers_indices_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Приблизительное количество кластеров: %d" % n_clusters_)
        print("Приблизительное количество точек шума: %d" % n_noise_)

    def getDb(self):
        print("Структура БД:")
        print(self.df)

    def getCenter(self):
        self.centres = self.df.loc[self.centroids]
        print(self.centres)

    def getPredict(self):
        self.predict['index'] = self.model.predict(self.predict.iloc[:].values)
        print(self.predict)
        self.set_predict = self.df[self.df['index'] == self.predict['index'].values[0]]
        print(self.set_predict)

    def getSaveAll(self):
        try:
            self.df.to_csv("./files/AffinityPropagation.csv", "@")
            self.centres.to_csv("./files/AffinityPropagationCenter.csv", "@")
            self.set_predict.to_csv("./files/AffinityPropagationPredict.csv", "@")
        except Exception:
            print("Не все файлы были сохранены. Пересмотрите их выполнение.", Exception)
