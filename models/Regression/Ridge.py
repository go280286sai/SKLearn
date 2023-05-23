"""
====================================================================
Ridge Линейный метод наименьших квадратов с регуляризацией
Проблема мультиколлинеарности возникает, когда входные признаки в модели
регрессии сильно коррелируют между собой, что может затруднить определение
влияния каждого признака на целевую переменную. В таких случаях использование
обычной линейной регрессии может привести к нестабильным и неправильным
оценкам коэффициентов.
====================================================================
class sklearn.linear_model.Ridge(alpha=1.0, *, fit_intercept=True,
copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)
"""
import pandas as pd
from sklearn import linear_model
from index import Index
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RidgeClass(Index):
    def getData(self):
        self.model = linear_model.Ridge(alpha=1.0)
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
        print("MSE:", MSE ** 0.5)
        # Влияние коэффициентов на прогноз.
        print(self.model.coef_)
        # Значение intercept будет числовым значением, которое представляет собой ожидаемое
        # значение y, когда все входные признаки равны нулю.
        print(self.model.intercept_)

    def getDb(self):
        print("Структура БД:")
        print(self.db)

    def getPredict(self):
        result = self.model.predict([self.args])
        print(result)
