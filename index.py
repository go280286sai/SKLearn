from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


class Index:
    def setData(self):
        self.data = pd.read_csv('./files/students.csv')

        # Перевод в цифровые метки
        self.coder = preprocessing.LabelEncoder()
        self.coder.fit(self.data['Sex'])
        self.data['Sex'] = self.coder.transform(self.data['Sex'])
        # Поиск аномалий
        # До проверки аномалий
        self.start = self.data.count()['Growth']

        self.data = self.anomaly(self.data['Growth'])
        self.data = self.anomaly(self.data['Shoe size'])
        self.data = self.anomaly(self.data['Weight'])

        # Удаляем не заполненные поля
        self.data.dropna()

        # После проверки аномалий
        self.finish = self.data.count()['Growth']

        # Сбрасываем индекс
        self.data = self.data.reset_index()

        # Выбираем нужные поля
        self.df = self.data[['Growth', 'Shoe size', 'Sex', 'Weight']].copy()

        # Выбираем данные из таблицы
        self.X = self.df.iloc[:, :-1].values
        self.y = self.df.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.4,
                                                                                random_state=42)

    def setPredict(self):
        self.args = [176, 42, 1]
        self.title = ['Growth', 'Shoe size', 'Sex']
        self.predict = pd.DataFrame([self.args],
                                    columns=self.title)

    def anomaly(self, fields):
        a = fields.quantile(0.25)
        b = fields.quantile(0.75)
        return self.data[(fields < b + 1.5 * (b - a)) & (fields > a - 1.5 * (b - a))]

    def getInfo(self):
        # До проверки аномалий
        print("Before start anomaly:", self.start)
        # После проверки аномалий
        print("After start anomaly:", self.finish)
        print('Delete anomaly:', self.start - self.finish)
        # Выводим сгруппированные данные
        # print(self.data.groupby([self.data['location'], self.data['loc']]).count()['id'])

        selector = ExtraTreesClassifier()
        result = selector.fit(self.df[self.df.columns], self.df['Weight'])
        result.feature_importances_
        features_table = pd.DataFrame(result.feature_importances_, index=self.df.columns,
                                      columns=['importance'])
        print(features_table)
        features_table.sort_values(by='importance', ascending=False)

