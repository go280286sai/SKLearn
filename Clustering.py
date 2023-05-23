from models.Clustering.AffinityPropagation import AffinityPropagationClass
from models.Clustering.AgglomerativeClustering import AgglomerativeClusteringClass
from models.Clustering.DBSCAN import DBSCANClass
from models.Clustering.KMean import KMeanClass
from models.Clustering.MeanShift import MeanShiftClass
from models.Clustering.SpectralClustering import SpectralClusteringClass

print("AffinityPropagation")
Affinity = AffinityPropagationClass()
# Применить настройки
Affinity.setData()
# Приминить модель
Affinity.getData()
# Получить центры кластеров
# Affinity.getCenter()
# Получить дополнительную информацию
# Affinity.getInfo()
# Вывести данные из БД
# Affinity.getDb()
# Установить данные для прогноза
# Affinity.setPredict()
# Получить прогноз
# Affinity.getPredict()
# Сохранить результаты в файл
# Affinity.getSaveAll()
print("------------------------------------------")

print("AgglomerativeClustering")
Agglomerative = AgglomerativeClusteringClass()
# Применить настройки
Agglomerative.setData()
# Получить центры кластеров
Agglomerative.getData()
# Вывести данные из БД
# Agglomerative.getDb()
# Сохранить результаты в файл
# Agglomerative.getSaveAll()
print("------------------------------------------")

print("DBSCAN")
DBSCAN = DBSCANClass()
# Применить настройки
DBSCAN.setData()
# Получить центры кластеров
DBSCAN.getData()
# Вывести данные из БД
# DBSCAN.getDb()
# Сохранить результаты в файл
# DBSCAN.getSaveAll()
print("------------------------------------------")

print("KMean")
KMean = KMeanClass()
# Применить настройки
KMean.setData()
# Получить центры кластеров
KMean.getData()
# Вывести данные из БД
# KMean.getDb()
# Установить данные для прогноза
# KMean.setPredict()
# Получить прогноз
# KMean.getPredict()
# Сохранить результаты в файл
# KMean.getSaveAll()
print("------------------------------------------")

print("MeanShift")
MeanShift = MeanShiftClass()
# Применить настройки
MeanShift.setData()
# Получить центры кластеров
MeanShift.getData()
# Вывести данные из БД
# MeanShift.getDb()
# Установить данные для прогноза
# MeanShift.setPredict()
# Получить прогноз
# MeanShift.getPredict()
# Сохранить результаты в файл
# MeanShift.getSaveAll()
print("------------------------------------------")

print("SpectralClustering")
SpectralClustering = SpectralClusteringClass()
# Применить настройки
SpectralClustering.setData()
# Получить центры кластеров
SpectralClustering.getData()
# Вывести данные из БД
# SpectralClustering.getDb()
# Установить данные для прогноза
# SpectralClustering.setPredict()
# Получить прогноз
# SpectralClustering.getPredict()
# Сохранить результаты в файл
# SpectralClustering.getSaveAll()
print("------------------------------------------")
