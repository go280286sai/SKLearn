"""
====================================================================
LogisticRegression является моделью логистической регрессии, которая
используется для решения задач бинарной классификации или многоклассовой
классификации. Она предсказывает вероятности принадлежности к определенным
классам на основе линейной комбинации входных признаков.
====================================================================
class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False,
tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0,
warm_start=False, n_jobs=None, l1_ratio=None)"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from index import Index
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LogisticRegressionClass(Index):
    def getData(self):
        self.model = LogisticRegression(random_state=0, max_iter=1000)
        self.model.fit(self.X, self.y)
        self.db = pd.DataFrame(self.X, columns=self.title)
        self.db['Analise'] = self.y
        self.db['Predict'] = y_predict = self.model.predict(self.X)
        score = self.model.score(self.X, self.y)
        print("R^2 score:", score)
        # Коэффициент от 0 до 1. Где значение 1 означает идеальное соответствие данных,
        # а значение 0 указывает на то, что модель не объясняет никакой вариации
        MAE = mean_absolute_error(self.db['Predict'], self.y)
        print("MAE:", MAE)
        MSE = mean_squared_error(self.db['Predict'], self.y)
        print("MSE:", MSE ** 0.5)
        accuracy = accuracy_score(self.y, y_predict)
        print("Точность модели: {:.2f}".format(accuracy))

    def getDb(self):
        print("Структура БД:")
        print(self.db)

    def getPredict(self):
        result = self.model.predict([self.args])
        result1 = self.model.predict_proba([self.args])
        print(result)
        print("Вероятность -:", result1[:, 0])
        print("Вероятность +:", result1[:, 1])



