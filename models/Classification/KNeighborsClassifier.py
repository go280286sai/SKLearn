"""
====================================================================
Модель KNeighborsRegressor из библиотеки scikit-learn (sklearn) используется
в задачах регрессии, когда требуется предсказывать непрерывное значение
целевой переменной на основе ближайших соседей.
====================================================================
class sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, *,
weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
metric_params=None, n_jobs=None)
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from index import Index
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class KNeighborsClassifierClass(Index):
    def getData(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(self.X_train, self.y_train)
        self.db = pd.DataFrame(self.X_test, columns=self.title)
        self.db['Analise'] = self.y_test
        self.db['Predict'] = self.model.predict(self.X_test)
        score = self.model.score(self.X_test, self.y_test)
        print("R^2 score:", score)
        # Коэффициент от 0 до 1. Где значение 1 означает идеальное соответствие данных,
        # а значение 0 указывает на то, что модель не объясняет никакой вариации
        MAE = mean_absolute_error(self.db['Predict'], self.y_test)
        print("MAE:", MAE)
        MSE = mean_squared_error(self.db['Predict'], self.y_test)
        print("MSE:", MSE**0.5)
        print(pd.crosstab(self.db['Analise'], self.db['Predict']))

    def getDb(self):
        print("Структура БД:")
        print(self.db)

    def getPredict(self):
        result = self.model.predict([self.args])
        print(result)

