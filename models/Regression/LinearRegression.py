"""
====================================================================
LineRegression Обычная линейная регрессия методом наименьших квадратов.
====================================================================
class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from index import Index
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LinearRegressionClass(Index):
    def getData(self):
        self.model = LinearRegression()
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