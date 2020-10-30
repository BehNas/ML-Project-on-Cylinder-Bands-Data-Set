import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def missing_data_percentage(df):
    features_nan = []
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            features_nan.append(column)
    for feature in features_nan:
        # print(feature, ":", df[feature].isnull().sum()/(size_df+1))
        print("{}:{}% missing values".format(feature, np.round(100*df[feature].isnull().mean(), 4)))


# We will replace by using median since there are outliers
def replace_numerical_missing_values(df, feature):
    median_value = df[feature].median()
    df[feature].fillna(median_value, inplace=True)


# We will replace missing values of categorical values with the values that is more frequent for each feature
def replace_categorical_missing_values(df, feature):
    values = df[feature].value_counts().keys().tolist()
    df[feature].fillna(values[0], inplace=True)


# we need to convert categorical values to numeric values in order to use them in ml algorithm
# Label Encoding method
def label_encoding(df, categorical_feature):
    label_encoder = LabelEncoder()
    for feature in categorical_feature:
        df[feature] = label_encoder.fit_transform(df[feature])
        df[feature] = pd.to_numeric(df[feature])


# one hot encoder method
def one_hot_encoding(df, categorical_feature):
    df.replace(('YES', 'NO'), (1, 0), inplace=True)
    df.replace(('BAND', 'NOBAND'), (1, 0), inplace=True)
    category = [feature for feature in categorical_feature if df[feature].dtype == 'O']
    return pd.get_dummies(df, columns=category, prefix=category)




