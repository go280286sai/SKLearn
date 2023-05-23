"""
====================================================================
MeanShift Его основная задача состоит в определении плотных областей
данных в многомерном пространстве.
====================================================================
class sklearn.cluster.MeanShift(*, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1,
cluster_all=True, n_jobs=None, max_iter=300)

"""
from sklearn.cluster import MeanShift
from index import Index


class MeanShiftClass(Index):
    def getData(self):
        self.model = MeanShift(bandwidth=5)
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
            self.df.to_csv("./files/MeanShift.csv", "@")
            self.set_predict.to_csv("./files/MeanShiftPredict.csv", "@")
        except Exception:
            print("Не все файлы были сохранены. Пересмотрите их выполнение.", Exception)
