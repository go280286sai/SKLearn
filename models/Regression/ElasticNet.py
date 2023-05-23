"""
====================================================================
Модель регрессии ElasticNet из библиотеки scikit-learn (sklearn) используется
в случаях, когда необходимо выполнить регуляризацию и совместно использовать
L1 (Lasso) и L2 (Ridge) регуляризацию. ElasticNet представляет собой комбинацию
обоих методов регуляризации и предлагает баланс между ними.
====================================================================
class sklearn.linear_model.ElasticNet(alpha=1.0, *, l1_ratio=0.5,
fit_intercept=True, precompute=False, max_iter=1000, copy_X=True,
tol=0.0001, warm_start=False, positive=False, random_state=None,
selection='cyclic')
"""
import pandas as pd
from sklearn.linear_model import ElasticNet
from index import Index
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ElasticNetClass(Index):
    def getData(self):
        self.model = ElasticNet(random_state=0)
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
