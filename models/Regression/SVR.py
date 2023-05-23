"""
====================================================================
SVR применяется, когда имеется набор данных с непрерывными целевыми
переменными, и требуется построить модель, которая может предсказывать
значения целевой переменной для новых наблюдений.
====================================================================
class sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False,
max_iter=-1)[source]
"""
import pandas as pd
from sklearn.svm import SVR
from index import Index
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class SVRClass(Index):
    def getData(self):
        self.model = SVR(kernel='linear')
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

    def getDb(self):
        print("Структура БД:")
        print(self.db)

    def getPredict(self):
        result = self.model.predict([self.args])
        print(result)

