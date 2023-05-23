"""
====================================================================
AgglomerativeClustering является одним из методов без учителя, используемых
в машинном обучении для группировки данных. Он относится к семейству
иерархической кластеризации, где объекты объединяются в иерархическую
структуру кластеров.
====================================================================
class sklearn.cluster.AgglomerativeClustering(n_clusters=2, *,
affinity='deprecated', metric=None, memory=None, connectivity=None,
compute_full_tree='auto', linkage='ward', distance_threshold=None,
compute_distances=False)
"""

from sklearn.cluster import AgglomerativeClustering
from index import Index


class AgglomerativeClusteringClass(Index):
    def getData(self):
        self.model = AgglomerativeClustering()
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
        self.df.to_csv("./files/AgglomerativeClustering.csv", "@")
