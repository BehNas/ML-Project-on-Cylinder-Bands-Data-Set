import numpy as np
import pandas as pd

missing_value_formats = ["None", "?", "NA", "n/a", "na", "--"]
# size_df = len(df.index)


def missing_data_percentage(df):
    features_nan = []
    df = df.replace(to_replace=missing_value_formats, value=np.nan)
    for feature in df.columns:
        df[feature] = df[feature].str.upper()
        if df[feature].isnull().sum() > 0:
            features_nan.append(feature)
    for feature in features_nan:
        # print(feature, ":", df[feature].isnull().sum()/(size_df+1))
        print("{}:{}% missing values".format(feature, np.round(100*df[feature].isnull().mean(), 4)))


def uppercase(df):
    for feature in df.columns:
        df[feature] = df[feature].str.upper()


def categorical_values(df):
    categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
    return categorical_features
