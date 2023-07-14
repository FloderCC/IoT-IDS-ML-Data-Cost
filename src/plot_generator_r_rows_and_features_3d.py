import csv

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

# running device MIPS
real_mips = 63250

# simulated device specs
# HP ProLiant ML110 G4, Intel Xeon 3075  / 4GB / 64GB (MIPS: 5320, Power: 93.7 - 135)
# ref: Khan, A. A., Zakarya, M., & Khan, R. (2019). Energy-aware dynamic resource management in elastic cloud datacenters. Simulation modelling practice and theory, 92, 82-99

simulated_mips = 5320
simulated_idle_energy = 93.7
simulated_busy_energy = 135

aux_constant = (simulated_busy_energy - simulated_idle_energy) / 100


# create the csv writer
f = open(f'plots/plot_summary.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(['Classifier', 'Dataset', 'MAX MCC value', 'MAX MCC sample size', 'MAX MCC number of features', 'MAX MCC energy consumed', 'CP MCC value', 'CP MCC sample size', 'CP MCC number of features', 'CP MCC energy consumed'])

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

def make_plot(classifier, dataset_no, metric):

    df = pd.read_csv(f'results_by_rows_and_features_d{dataset_no}.csv')

    df['Training time'] = df['Training time'] * (real_mips / simulated_mips)
    df['Energy consumption'] = df['Training time'] * (simulated_idle_energy + aux_constant * df['TR-CPU%'])


    df = df[df['Classifier'] == classifier]

    x = df['Dataset %'].values
    y = df['Number of features'].values
    z = df[metric].values

    plt.rcParams['figure.figsize'] = [8.4, 6]
    plt.rcParams['font.size'] = 16

    fig = plt.figure()
    ax = Axes3D(fig)

    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
    cbar = fig.colorbar(surf, shrink=0.6, aspect=6, pad=0.1)
    cbar.ax.set_position(cbar.ax.get_position().translated(-0.038, 0))

    # Set labels and title
    ax.set_xlabel('Number of instances (%)')
    ax.set_ylabel('Number of features')


    # Find the indices of the maximum MCC value
    max_mcc_index = np.argmax(z)
    # Retrieve the corresponding coordinates
    max_mcc_x = x[max_mcc_index]
    max_mcc_y = y[max_mcc_index]
    max_mcc_z = z[max_mcc_index]
    max_mcc_energy = df["Energy consumption"].values[max_mcc_index]

    # Find the threshold for 95% of the highest MCC value
    threshold = 0.985 * max_mcc_z
    # Find the index of the MCC value that reaches the threshold
    index_threshold = np.where(z >= threshold)[0][0]
    # Retrieve the corresponding coordinates
    threshold_x = x[index_threshold]
    threshold_y = y[index_threshold]
    threshold_z = z[index_threshold]
    threshold_energy = df["Energy consumption"].values[index_threshold]

    # Remove white spaces on top and left
    fig.subplots_adjust(left=0, bottom=0, top=1, right=1)
    ax.set_position([0, 0, 0.7, 1])

    if metric == 'MCC':
        ax.set_zlabel(metric, labelpad=14)
        plt.savefig(f'plots/plot dataset {dataset_no-4}, {metric}, {rename_classifier(classifier)}.pdf')
        writer.writerow([rename_classifier(classifier), dataset_no-4, round(max_mcc_z, 5), max_mcc_x, max_mcc_y, max_mcc_energy, round(threshold_z, 5), threshold_x, threshold_y, threshold_energy])
    else:
        ax.set_zlabel(metric, labelpad=26)
        ax.zaxis.set_tick_params(pad=12)
        plt.savefig(f'plots/plot dataset {dataset_no-4}, energy consumption, {rename_classifier(classifier)}.pdf')

        # plt.show()
        # exit()




classifier_list = ['RFR', 'DTR', 'ADB', 'LR', 'MLP1', 'MLP2', 'MLP3']
dataset_no_list = [5, 6, 7, 8]
metric_list = ['MCC', 'Energy consumption']

for classifier in classifier_list:
    for dataset_no in dataset_no_list:
        for metric in metric_list:
            make_plot(classifier, dataset_no, metric)

# close the file
f.close()