import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

missing_value_formats = ["None", "?", "NA", "n/a", "na", "--"]
# size_df = len(df.index)


def missing_data_percentage(df):
    features_nan = []
    df = df.replace(to_replace=missing_value_formats, value=np.nan)
    print(df)
    for column in df.columns:
        df[column] = df[column].str.upper()
        if df[column].isnull().sum() > 0:
            features_nan.append(column)
    for feature in features_nan:
        # print(feature, ":", df[feature].isnull().sum()/(size_df+1))
        print("{}:{}% missing values".format(feature, np.round(100*df[feature].isnull().mean(), 4)))


def uppercase(df):
    for column in df.columns:
        df[column] = df[column].str.upper()
    return df


def convert_to_nan(df):
    df = df.replace(to_replace=missing_value_formats, value=np.nan)
    return df


def categorical_values(df):
    categorical_features = [column for column in df.columns if df[column].dtype == 'O']
    return categorical_features


def numeric_values(df):
    numeric_features = [column for column in df.columns if df[column].dtype != 'O']
    return numeric_features


def convert_to_numeric(df):
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='ignore')


# find discrete values
def discrete_values(df):
    return [feature for feature in numeric_values(df) if len(df[feature].unique()) < 10]


# find continuous values
def continuous_values(df):
    return [feature for feature in numeric_values(df) if feature not in discrete_values(df)]


# analyse the continuous values by creating histograms to understand the distribution
def distribution_histogram(df):
    size_df = len(df.index)
    for feature in continuous_values(df):
        df[feature].hist(bins=10)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(feature)
        plt.show()


# outliers
def box_plot_distribution(df):
    for feature in continuous_values(df):
        if 0 in df[feature].unique():
            pass
        else:
            df[feature] = np.log(df[feature])
            df.boxplot(column=feature)
            plt.ylabel(feature)
            plt.title(feature)
            plt.show()


# pair-plot for numeric values
def pair_plot(df, m, n):
    sns.pairplot(df, hue='band type', kind='scatter', vars=continuous_values(df)[m: n],
                                                          plot_kws=dict(alpha=0.5),
                                                          diag_kws=dict(alpha=0.5))
    plt.show()