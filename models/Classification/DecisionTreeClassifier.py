"""
====================================================================
Модель DecisionTreeRegressor из модуля sklearn.tree используется для
задач регрессии, то есть для прогнозирования непрерывных числовых
значений на основе входных признаков. Она строит дерево решений,
которое разбивает пространство признаков на несколько регионов и в
каждом регионе предсказывает числовое значение целевой переменной.
====================================================================
class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best',
max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
class_weight=None, ccp_alpha=0.0)
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from index import Index


class DecisionTreeClassifierClass(Index):
    def getData(self):
        self.model = DecisionTreeClassifier(random_state=0)
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


