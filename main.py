import argparse
import csv
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from Data_preprocessing import data_cleanup

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
    # df.head()

    return df






def main(input_dir, output_file):
    data_frame = load_data(input_dir)
    # data_cleanup.missing_data_percentage(data_frame)
    column_title_row = ['index', ' ' * 2 + 'Acuracy', ' ' * 2 + 'Class', ' ' * 2 + 'Md5', ' ' * 2 + 'blur']
    with open(output_file, 'w', encoding="utf-8") as csvfile:
          testwriter = csv.writer(csvfile, delimiter=',', lineterminator="\n")
          testwriter.writerow(column_title_row)
    #     u = len(sd_img_path_list)
    #     for i in range(len(sd_img_path_list)):
    #         blur = calculating_blur(sd_img_path_list[i])
    #         testwriter.writerow([i + 1, sd_img_path_list[i], class_list[i], images_md5[i], blur])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cylinder band classifier")
    parser.add_argument("input_dir", help="path to the Data Set")
    parser.add_argument("output_file", default="report.csv", help="csv file name")
    args = parser.parse_args()
    main(args.input_dir, args.output_file)
