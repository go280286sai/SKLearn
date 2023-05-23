"""
====================================================================
Класс MLPRegressor из модуля sklearn.neural_network используется для
реализации нейронных сетей с многослойной перцептронной архитектурой
(MLP) в задачах регрессии. MLPRegressor позволяет создавать модели,
которые могут обрабатывать нелинейные зависимости между входными
признаками и целевыми переменными.
====================================================================
class sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(100,),
activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto',
learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
max_iter=200, shuffle=True, random_state=None, tol=0.0001,
verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
"""
import pandas as pd
from sklearn.neural_network import MLPRegressor
from index import Index
from sklearn.metrics import mean_absolute_error, mean_squared_error

class MLPRegressorClass(Index):
    def getData(self):
        self.model = MLPRegressor(random_state=1, max_iter=500)
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
