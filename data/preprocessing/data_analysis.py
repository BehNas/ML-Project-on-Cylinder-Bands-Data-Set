import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

missing_value_formats = ["None", "?", "NA", "n/a", "na", "--"]
# size_df = len(df.index)


# Uppercase all categorical and convert all numbers to numeric
def clean(df, feature):
    df[feature] = df[feature].str.upper()
    df[feature] = pd.to_numeric(df[feature], errors='ignore')
    if df[feature].nunique() == 1 or feature == 'cylinder number':
        del df[feature]


def convert_to_nan(df):
    return df.replace(to_replace=missing_value_formats, value=np.nan)


def categorical_numeric_split(df):
    categorical_feature = []
    numeric_feature = []
    for feature in df.columns:
        if df[feature].dtype == 'O':
            categorical_feature.append(feature)
        else:
            numeric_feature.append(feature)
    return categorical_feature, numeric_feature


# find discrete values
def discrete_values(df, numeric):
    return [feature for feature in numeric if len(df[feature].unique()) < 10]


# find continuous values
def continuous_values(df, numeric):
    return [feature for feature in numeric if len(df[feature].unique()) > 10]


# analyse the continuous values by creating histograms to understand the distribution
def distribution_histogram(df, continuous):
    size_df = len(df.index)
    for feature in continuous:
        df[feature].hist(bins=10)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(feature)
        plt.show()


# outliers
def box_plot_distribution(df, continuous):
    for feature in continuous:
        if 0 in df[feature].unique():
            pass
        else:
            df[feature] = np.log(df[feature])
            df.boxplot(column=feature)
            plt.ylabel(feature)
            plt.title(feature)
            plt.show()


# pair-plot for numeric values
def pair_plot(df, continuous, m, n):
    sns.pairplot(df, hue='band type', kind='scatter', vars=continuous[m: n],
                 plot_kws=dict(alpha=0.5),
                 diag_kws=dict(alpha=0.5))
    plt.show()
