"""
====================================================================
K-Means
====================================================================
class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init='warn',
max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')[source]
"""
from sklearn.cluster import KMeans
from index import Index


class KMeanClass(Index):
    def getData(self):
        self.model = KMeans(n_clusters=3, random_state=0, n_init="auto")
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
            self.df.to_csv("./files/KMean.csv", "@")
            self.set_predict.to_csv("./files/KMeanPredict.csv", "@")
        except Exception:
            print("Не все файлы были сохранены. Пересмотрите их выполнение.", Exception)
