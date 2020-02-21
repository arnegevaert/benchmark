from os import path
from lib import util
import pandas as pd
from sklearn.model_selection import train_test_split


class GermanCredit:
    def __init__(self, test_size=0.2,
                 data_location=path.join(path.dirname(__file__), "../../data")):
        df = pd.read_csv(path.join(data_location, "german_credit", "german_credit.csv"), delimiter=",")
        class_name = "default"
        type_features, features_type = util.recognize_features_type(df)

        discrete = ['installment_as_income_perc', 'present_res_since', 'credits_this_bank', 'people_under_maintenance']
        discrete, continuous = util.set_discrete_continuous(
            df.columns, type_features, class_name, discrete, continuous=None)

        columns_tmp = list(df.columns)
        columns_tmp.remove(class_name)
        self.idx_features = {i: col for i, col in enumerate(columns_tmp)}

        df_le, label_encoder = util.label_encode(df, discrete)

        X = df_le.loc[:, df_le.columns != class_name].values
        y = df_le[class_name].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)


if __name__ == '__main__':
    gc = GermanCredit()
