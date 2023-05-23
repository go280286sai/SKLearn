from models.Classification.DecisionTreeClassifier import DecisionTreeClassifierClass
from models.Classification.GradientBoostingClassifier import GradientBoostingClassifierClass
from models.Classification.KNeighborsClassifier import KNeighborsClassifierClass
from models.Classification.LogisticRegression import LogisticRegressionClass
from models.Classification.RandomForestClassifier import RandomForestClassifierClass

print("DecisionTreeClassifier")
Decision = DecisionTreeClassifierClass()
# Применить настройки
Decision.setData()
# Установить данные для прогноза
Decision.setPredict()
# Приминить модель
Decision.getData()
# Получить дополнительную информацию
# Decision.getInfo()
# Вывести данные из БД
# Decision.getDb()
# Получить прогноз
# Decision.getPredict()
print("------------------------------------------")

print("GradientBoostingClassifier")
Gradient = GradientBoostingClassifierClass()
# Применить настройки
Gradient.setData()
# Установить данные для прогноза
Gradient.setPredict()
# Приминить модель
Gradient.getData()
# Получить дополнительную информацию
# Gradient.getInfo()
# Вывести данные из БД
# Gradient.getDb()
# Получить прогноз
# Gradient.getPredict()
print("------------------------------------------")

print("KNeighborsClassifier")
KNeighbors = KNeighborsClassifierClass()
# Применить настройки
KNeighbors.setData()
# Установить данные для прогноза
KNeighbors.setPredict()
# Приминить модель
KNeighbors.getData()
# Получить дополнительную информацию
# KNeighbors.getInfo()
# Вывести данные из БД
# KNeighbors.getDb()
# Получить прогноз
# KNeighbors.getPredict()
print("------------------------------------------")

print("RandomForestClassifier")
RandomForest = RandomForestClassifierClass()
# Применить настройки
RandomForest.setData()
# Установить данные для прогноза
RandomForest.setPredict()
# Приминить модель
RandomForest.getData()
# Получить дополнительную информацию
# RandomForest.getInfo()
# Вывести данные из БД
# RandomForest.getDb()
# Получить прогноз
# RandomForest.getPredict()
print("------------------------------------------")

print("LogisticRegression")
Logistic = LogisticRegressionClass()
# Применить настройки
Logistic.setData()
# Установить данные для прогноза
Logistic.setPredict()
# Приминить модель
Logistic.getData()
# Получить дополнительную информацию
# Logistic.getInfo()
# Вывести данные из БД
# Logistic.getDb()
# Получить прогноз
# Logistic.getPredict()
print("------------------------------------------")