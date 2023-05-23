"""
====================================================================
Модель DecisionTreeRegressor из модуля sklearn.tree используется для
задач регрессии, то есть для прогнозирования непрерывных числовых
значений на основе входных признаков. Она строит дерево решений,
которое разбивает пространство признаков на несколько регионов и в
каждом регионе предсказывает числовое значение целевой переменной.
====================================================================
class sklearn.tree.DecisionTreeRegressor(*, criterion='squared_error',
splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0)
"""
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from index import Index
from sklearn.metrics import mean_absolute_error, mean_squared_error


class DecisionTreeRegressorClass(Index):
    def getData(self):
        self.model = DecisionTreeRegressor(max_depth=2, random_state=0)
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
        print(cross_val_score(self.model, self.X, self.y, cv=10))

    def getDb(self):
        print("Структура БД:")
        print(self.db)

    def getPredict(self):
        result = self.model.predict([self.args])
        print(result)
