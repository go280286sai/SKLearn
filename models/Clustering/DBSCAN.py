"""
====================================================================
DBSCAN это алгоритм кластеризации, который используется
для выявления групп объектов в пространстве на основе их плотности.
Плотность определяется как количество объектов в заданном радиусе вокруг
данного объекта.
====================================================================
class sklearn.cluster.DBSCAN(eps=0.5, *, min_samples=5, metric='euclidean',
metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
"""
from sklearn.cluster import DBSCAN
from index import Index


class DBSCANClass(Index):
    def getData(self):
        self.model = DBSCAN(eps=2, min_samples=2)
        self.model.fit(self.X)
        self.df['index'] = labels = self.model.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Приблизительное количество кластеров: %d" % n_clusters_)
        print("Приблизительное количество точек шума: %d" % n_noise_)

    def getDb(self):
        print("Структура БД:")
        print(self.df)

    def getSaveAll(self):
        self.df.to_csv("./files/DBSCAN.csv", '@')
