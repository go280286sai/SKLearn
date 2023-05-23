"""
====================================================================
SpectralClustering Он основан на спектральном анализе матрицы сходства между объектами данных.
====================================================================
class sklearn.cluster.SpectralClustering(n_clusters=8, *, eigen_solver=None, n_components=None,
random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol='auto',
assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False)
"""
from sklearn.cluster import SpectralClustering
from index import Index


class SpectralClusteringClass(Index):
    def getData(self):
        self.model = SpectralClustering(n_clusters=3, assign_labels='discretize', random_state=0)
        self.model.fit(self.X)
        self.df['index'] = labels = self.model.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Приблизительное количество кластеров: %d" % n_clusters_)
        print("Приблизительное количество точек шума: %d" % n_noise_)

    def getDb(self):
        print("Структура БД:")
        print(self.df)

    def getPredict(self):
        self.predict['index'] = self.model.predict(self.predict.iloc[:].values)
        print(self.predict)
        self.set_predict = self.df[self.df['index'] == self.predict['index'].values[0]]
        print(self.set_predict)

    def getSaveAll(self):
        try:
            self.df.to_csv("./files/SpectralClustering.csv", "@")
            self.set_predict.to_csv("./files/KMeanPredict.csv", "@")
        except Exception:
            print("Не все файлы были сохранены. Пересмотрите их выполнение.", Exception)

