import csv
from statistics import mean

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from scipy.stats import spearmanr

# running device MIPS
real_mips = 63250

# simulated device specs
# HP ProLiant ML110 G4, Intel Xeon 3075  / 4GB / 64GB (MIPS: 5320, Power: 93.7 - 135)
# ref: Khan, A. A., Zakarya, M., & Khan, R. (2019). Energy-aware dynamic resource management in elastic cloud datacenters. Simulation modelling practice and theory, 92, 82-99

simulated_mips = 5320
simulated_idle_energy = 93.7
simulated_busy_energy = 135

aux_constant = (simulated_busy_energy - simulated_idle_energy) / 100


def rename_classifier(name):
    # 'RFR', 'DTR', 'ADB', 'LR'
    if name == 'RFR':
        return 'RF'
    if name == 'DTR':
        return 'DT'
    if name == 'ADB':
        return 'AdaBoost'
    if name == 'LR':
        return 'LogReg'
    return name


def get_coefficient(classifier, dataset_no, col1, col2):

    df = pd.read_csv(f'results_by_rows_and_features_d{dataset_no}.csv')
    df = df.loc[df['Classifier'] == classifier]

    df['Training time'] = df['Training time'] * (real_mips / simulated_mips)
    df['Energy consumption'] = df['Training time'] * (simulated_idle_energy + aux_constant * df['TR-CPU%'])

    correlation, p_value = spearmanr(df[col1], df[col2])
    print(f"Spearman's correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")

    return str(correlation)
    # return str(correlation) + ' (' + str(p_value) + ')'


f = open('plots/statistical_summary.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(['Classifier', 'Dataset', 'Instances VS energy SRCC', 'Features VS energy SRCC', 'Instances VS MCC SRCC', 'Features VS MCC SRCC', 'MCC VS energy SRCC'])


classifier_list = ['RFR', 'DTR', 'ADB', 'LR', 'MLP1', 'MLP2', 'MLP3']
dataset_no_list = [5, 6, 7, 8]
metric_list = ['MCC', 'Energy consumption']

for classifier in classifier_list:
    for dataset_no in dataset_no_list:
        writer.writerow([rename_classifier(classifier), dataset_no-4,
                         get_coefficient(classifier, dataset_no, 'Dataset %', 'Energy consumption'),
                         get_coefficient(classifier, dataset_no, 'Number of features', 'Energy consumption'),
                         get_coefficient(classifier, dataset_no, 'Dataset %', 'MCC'),
                         get_coefficient(classifier, dataset_no, 'Number of features', 'MCC'),
                         get_coefficient(classifier, dataset_no, 'MCC', 'Energy consumption')
                         ])

# close the file
f.close()

# df = pd.read_csv('plots/statistical_summary.csv')
#
# print('Total: ', len(df))
# print('Total Instances energy C > 0.8: ', len(df.loc[df['Instances VS energy SRCC'] > 0.8]))
# print('Total Instances energy C AVG: ', mean(df['Instances VS energy SRCC']))
#
# print('Total Instances MCC C AVG: ', mean(df['Instances VS MCC SRCC']))
#
#
# print('Total Features energy C > 0.8: ', len(df.loc[df['Features VS energy SRCC'] > 0.8]))
# print('Total Features energy C AVG: ', mean(df['Features VS energy SRCC']))
#
# print('Total Features MCC C AVG: ', mean(df['Features VS MCC SRCC']))

# Total:  28
# Total Instances energy C > 0.8:  16
# Total Instances energy C AVG:  0.7904384295284544
# Total Instances MCC C AVG:  -0.04702289075128353
# Total Features energy C > 0.8:  1
# Total Features energy C AVG:  0.38600090757596023
# Total Features MCC C AVG:  0.20900121698619428
#
# Process finished with exit code 0

