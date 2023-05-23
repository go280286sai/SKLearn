# SKLearn
![](./files/Scikit_learn_logo.png)

Подключаем файл

        self.data = pd.read_csv('./files/students.csv')
        
### Перевод в цифровые метки
        self.coder = preprocessing.LabelEncoder()
        self.coder.fit(self.data['Sex'])
        self.data['Sex'] = self.coder.transform(self.data['Sex'])

## Поиск аномалий
### До проверки аномалий
        self.start = self.data.count()['Growth']

        self.data = self.anomaly(self.data['Growth'])
        self.data = self.anomaly(self.data['Shoe size'])
        self.data = self.anomaly(self.data['Weight'])

### Удаляем не заполненные поля
        self.data.dropna()

### После проверки аномалий
        self.finish = self.data.count()['Growth']

### Сбрасываем индекс
        self.data = self.data.reset_index()

### Выбираем нужные поля
        self.df = self.data[['Growth', 'Shoe size', 'Sex', 'Weight']].copy()

### Выбираем данные из таблицы
        self.X = self.df.iloc[:, :-1].values
        self.y = self.df.iloc[:, -1].values
### Делим базу на тренировочную и тестовую
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.4,
                                                                                random_state=42)
### Создаем таблицу те прогнозирования
    def setPredict(self):
        self.args = [176, 42, 1]
        self.title = ['Growth', 'Shoe size', 'Sex']
        self.predict = pd.DataFrame([self.args],
                                    columns=self.title)
## Функция для поиска аномальных данных и исключение их
    def anomaly(self, fields):
        a = fields.quantile(0.25)
        b = fields.quantile(0.75)
        return self.data[(fields < b + 1.5 * (b - a)) & (fields > a - 1.5 * (b - a))]

## Фунция для получения мнформации

    def getInfo(self):
### До проверки аномалий
        print("Before start anomaly:", self.start)
### После проверки аномалий
        print("After start anomaly:", self.finish)
        print('Delete anomaly:', self.start - self.finish)
### Выводим сгруппированные данные
        print(self.data.groupby([self.data['location'], self.data['loc']]).count()['id'])

### Определение и вывод факторов влияющих на результат
        selector = ExtraTreesClassifier()
        result = selector.fit(self.df[self.df.columns], self.df['Weight'])
### где, self.df.columns - все колонки, self.df['Weight'] - прогнозируемая
        result.feature_importances_
        features_table = pd.DataFrame(result.feature_importances_, index=self.df.columns,
                                      columns=['importance'])
        print(features_table)
        features_table.sort_values(by='importance', ascending=False)

# Модели регрессии

    ./Regression.py

## DecisionTreeRegressor
Для прогнозирования непрерывных числовых
значений на основе входных признаков. Она строит дерево решений,
которое разбивает пространство признаков на несколько регионов и в
каждом регионе предсказывает числовое значение целевой переменной.

    Decision = DecisionTreeRegressorClass()
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
    Decision.getPredict()
    print("------------------------------------------")

## ElasticNet
Используется в случаях, когда необходимо выполнить регуляризацию и совместно использовать
L1 (Lasso) и L2 (Ridge) регуляризацию. ElasticNet представляет собой комбинацию
обоих методов регуляризации и предлагает баланс между ними.

    Elastic = ElasticNetClass()
    # Применить настройки
    Elastic.setData()
    # Установить данные для прогноза
    Elastic.setPredict()
    # Приминить модель
    Elastic.getData()
    # Получить дополнительную информацию
    # Elastic.getInfo()
    # Вывести данные из БД
    # Elastic.getDb()
    # Получить прогноз
    Elastic.getPredict()
    print("------------------------------------------")

## GradientBoostingRegressor
Используется для задач регрессии, то есть для прогнозирования непрерывных числовых
значений на основе входных признаков. Она строит дерево решений,
которое разбивает пространство признаков на несколько регионов и в
каждом регионе предсказывает числовое значение целевой переменной.
    
    Gradient = GradientBoostingRegressorClass()
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
    Gradient.getPredict()
    print("------------------------------------------")

## KNeighborsRegressor
Используется в задачах регрессии, когда требуется предсказывать непрерывное значение
целевой переменной на основе ближайших соседей.

    KNeighbors = KNeighborsRegressorClass()
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
    KNeighbors.getPredict()
    print("------------------------------------------")

## Lasso
Оценивает разреженные коэффициенты. Таким образом, модель Lasso часто используется, когда требуется отбор
признаков, регуляризация или интерпретируемость модели.

    Lasso = LassoClass()
    # Применить настройки
    Lasso.setData()
    # Установить данные для прогноза
    Lasso.setPredict()
    # Приминить модель
    Lasso.getData()
    # Получить дополнительную информацию
    # Lasso.getInfo()
    # Вывести данные из БД
    # Lasso.getDb()
    # Получить прогноз
    Lasso.getPredict()
    print("------------------------------------------")

## LinearRegression
Обычная линейная регрессия методом наименьших квадратов

    Linear = LinearRegressionClass()
    # Применить настройки
    Linear.setData()
    # Установить данные для прогноза
    Linear.setPredict()
    # Приминить модель
    Linear.getData()
    # Получить дополнительную информацию
    # Linear.getInfo()
    # Вывести данные из БД
    # Linear.getDb()
    # Получить прогноз
    Linear.getPredict()
    print("------------------------------------------")

## MLPRegressor
используется для реализации нейронных сетей с многослойной перцептронной архитектурой
(MLP) в задачах регрессии. MLPRegressor позволяет создавать модели,
которые могут обрабатывать нелинейные зависимости между входными
признаками и целевыми переменными.

    MLP = MLPRegressorClass()
    # Применить настройки
    MLP.setData()
    # Установить данные для прогноза
    MLP.setPredict()
    # Приминить модель
    MLP.getData()
    # Получить дополнительную информацию
    # MLP.getInfo()
    # Вывести данные из БД
    # MLP.getDb()
    # Получить прогноз
    MLP.getPredict()
    print("------------------------------------------")

## RandomForestRegressor
Используется в случаях, когда требуется построить регрессионную
модель на основе ансамбля случайных лесов.

    RandomForest = RandomForestRegressorClass()
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
    RandomForest.getPredict()
    print("------------------------------------------")

## Ridge
Линейный метод наименьших квадратов с регуляризацией
Проблема мультиколлинеарности возникает, когда входные признаки в модели
регрессии сильно коррелируют между собой, что может затруднить определение
влияния каждого признака на целевую переменную. В таких случаях использование
обычной линейной регрессии может привести к нестабильным и неправильным
оценкам коэффициентов.

    Ridge = RidgeClass()
    # Применить настройки
    Ridge.setData()
    # Установить данные для прогноза
    Ridge.setPredict()
    # Приминить модель
    Ridge.getData()
    # Получить дополнительную информацию
    # Ridge.getInfo()
    # Вывести данные из БД
    # Ridge.getDb()
    # Получить прогноз
    Ridge.getPredict()
    print("------------------------------------------")

## SVR
Применяется, когда имеется набор данных с непрерывными целевыми
переменными, и требуется построить модель, которая может предсказывать
значения целевой переменной для новых наблюдений.

    SVR = SVRClass()
    # Применить настройки
    SVR.setData()
    # Установить данные для прогноза
    SVR.setPredict()
    # Приминить модель
    SVR.getData()
    # Получить дополнительную информацию
    # SVR.getInfo()
    # Вывести данные из БД
    # SVR.getDb()
    # Получить прогноз
    SVR.getPredict()
    print("------------------------------------------")

## Где:
    def getData(self):
### Определение модели
        self.model = SVR(kernel='linear')
### Определяем данные
        self.model.fit(self.X_train, self.y_train)
### Создаем новую таблицу и заносим результат
        self.db = pd.DataFrame(self.X_test, columns=self.title)
        self.db['Analise'] = self.y_test
        self.db['Predict'] = self.model.predict(self.X_test)
### Коэффициент score от 0 до 1. Где значение 1 означает идеальное соответствие данных, а значение 0 указывает на то, что модель не объясняет никакой вариации
        score = self.model.score(self.X_test, self.y_test)
        print("R^2 score:", score)
### Измеряем среднюю разницу между прогнозируемыми значениями и фактическими значениями
        MAE = mean_absolute_error(self.db['Predict'], self.y_test)
        print("MAE:", MAE)
### Измеряем среднее значение суммы квадратов каждой разницы между оценочным значением и истинным значением
        MSE = mean_squared_error(self.db['Predict'], self.y_test)
        print("MSE:", MSE**0.5)
### Выводим таблицу
    def getDb(self):
        print("Структура БД:")
        print(self.db)
### Выводим прогнозируемое значение
    def getPredict(self):
        result = self.model.predict([self.args])
        print(result)