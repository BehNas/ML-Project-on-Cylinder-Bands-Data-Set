import argparse
import csv
import pandas as pd
from data.preprocessing import data_analysis
from data.preprocessing import missing_data_imputation
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
from sklearn.metrics import f1_score


def load_data(path_data):
    with open(path_data) as fp:
        data = [line.split(maxsplit=40) for line in fp]
    total_data = []
    for i in range(len(data)):
        d = data[i]
        if d:
            temp = d[0].split(',')
            total_data.append(temp)
    df = pd.DataFrame(total_data)
    df.columns = ['timestamp', 'cylinder number', 'customer', 'job number', 'grain screened', 'ink color',
                  'proof on ctd ink', 'blade mfg', 'cylinder division', 'paper type', 'ink type', 'direct steam',
                  'solvent type', 'type on cylinder', 'press type', 'press', 'unit number', 'cylinder size',
                  'paper mill location', 'plating tank', 'proof cut', 'viscosity', 'caliper', 'ink temperature',
                  'humifity', 'roughness', 'blade pressure', 'varnish pct', 'press speed', 'ink pct', 'solvent pct',
                  'ESA Voltage', 'ESA Amperage', 'wax', 'hardener', 'roller durometer', 'current density',
                  'anode space ratio', 'chrome content', 'band type']

    pd.pandas.set_option('display.max_columns', None)
    return df


def main(input_dir, output_file):
    data_frame = load_data(input_dir)
    data_frame = data_analysis.convert_to_nan(data_frame)
    for feature in data_frame.columns:
        data_analysis.clean(data_frame, feature)
    missing_data_imputation.missing_data_percentage(data_frame)
    categorical_feature, numeric_feature = data_analysis.categorical_numeric_split(data_frame)
    continuous_feature = data_analysis.continuous_values(data_frame, numeric_feature)
    # data_analysis.distribution_histogram(data_frame, continuous_feature)
    # data_analysis.box_plot_distribution(data_frame, continuous_feature)
    # data_analysis.pair_plot(data_frame, continuous_feature, 3, 7)
    for feature in numeric_feature:
        if data_frame[feature].isnull().sum() > 0:
            missing_data_imputation.replace_numerical_missing_values(data_frame, feature)
    for feature in categorical_feature:
        missing_data_imputation.replace_categorical_missing_values(data_frame, feature)

    data_frame = missing_data_imputation.one_hot_encoding(data_frame, categorical_feature)
    # missing_data_imputation.label_encoding(data_frame, categorical_feature)
    df_y = data_frame.iloc[:, data_frame.columns.get_loc('band type')]
    df_x = data_frame.drop('band type', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, stratify=df_y, test_size=0.20)
    # clf = tree.DecisionTreeClassifier()
    # clf = svm.SVC()
    clf = ensemble.RandomForestClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_probs = clf.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1])
    auc = roc_auc_score(y_test, y_probs[:, 1])
    f1 = f1_score(y_test, y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print('AUC: %.3f' % auc)
    print('f1=%.3f' % f1)
    pyplot.plot(fpr, tpr, marker='.', label='Band Type')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs[:, 1])
    pyplot.plot(recall, precision, linestyle='--', label='Type Band')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.show()
    column_title_row = ['Classifier model', ' ' * 2 + 'Accuracy', ' ' * 2 + 'AUC', ' ' * 2 + 'F1',
                        ' ' * 2 + 'Precision', ' ' * 2 + 'Recall', ' ' * 2 + 'False Positive Rate',
                        ' ' * 2 + 'True Positive Rate']
    with open(output_file, 'w', encoding="utf-8") as csvfile:
        testwriter = csv.writer(csvfile, delimiter=',', lineterminator="\n")
        testwriter.writerow(column_title_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cylinder band classifier")
    parser.add_argument("input_dir", help="path to the Data Set")
    parser.add_argument("output_file", default="report.csv", help="csv file name")
    args = parser.parse_args()
    main(args.input_dir, args.output_file)
