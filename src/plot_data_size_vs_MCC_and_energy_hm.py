"""File description:

Script for generating the head maps between instances and features vs MCC or energy. This script also generate the top
scores summary file, and the Readme file for the image folder

Note that the variable 'real_mips' should represent the number of MIPS available on the computer where the experiment
was executed.
"""

import csv
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata
import seaborn as sns


# setup for the energy consumption model

# running device MIPS
real_mips = 63250
# simulated device specs
# HP ProLiant ML110 G4, Intel Xeon 3075  / 4GB / 64GB (MIPS: 5320, Power: 93.7 - 135)
# ref: Khan, A. A., Zakarya, M., & Khan, R. (2019). Energy-aware dynamic resource management in elastic cloud datacenters. Simulation modelling practice and theory, 92, 82-99
simulated_mips = 5320
simulated_idle_energy = 93.7
simulated_busy_energy = 135
aux_constant = (simulated_busy_energy - simulated_idle_energy) / 100

# create the Readme file
readme_file = open(f'plots/README.md', 'w')
readme_file.write('# Plots data size vs MCC and energy\n')

# create the csv writer
ts_file = open(f'results/top scores summary.csv', 'w', newline='')
ts_writer = csv.writer(ts_file)
ts_writer.writerow(['Classifier', 'Dataset', 'MAX MCC value', 'MAX MCC sample size', 'MAX MCC number of features', 'MAX MCC energy consumed', 'CP MCC value', 'CP MCC sample size', 'CP MCC number of features', 'CP MCC energy consumed'])

def make_plot(dataset, classifier, metric):

    # loading the results
    df = pd.read_csv(f'results/results_by_rows_and_features.csv')

    # filtering the results by dataset & classifier & AVG of sampling seeds
    df = df[df['Dataset name'] == dataset]
    df = df[df['Classifier'] == classifier]
    df = df[df['Random seed'] == 'AVG']

    # computing and adding the energy consumption
    df['Training time'] = df['Training time'] * (real_mips / simulated_mips)
    df['Energy consumption'] = df['Training time'] * (simulated_idle_energy + aux_constant * df['TR-CPU%'])

    # plot configuration

    plt.rcParams['figure.figsize'] = [8.4, 6]
    plt.rcParams['font.size'] = 16

    # Assuming you have a DataFrame named 'df'
    pivot_table = df.pivot_table(index='Dataset %', columns='Number of features', values=metric)

    # Create the heatmap
    heatmap = sns.heatmap(pivot_table, cmap='Blues' if metric == 'MCC' else 'RdYlGn_r', cbar_kws={'label': metric if metric == 'MCC' else metric+' (J)'})

    # Set labels and title
    plt.xlabel('Number of input features', labelpad=7)
    plt.ylabel('Number of instances (%)')

    # Find the indices of the maximum MCC value
    x = df['Dataset %'].values
    y = df['Number of features'].values
    z = df[metric].values

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
    # fig.subplots_adjust(left=0, bottom=0, top=1, right=1)
    # ax.set_position([0, 0, 0.65, 1])
    # cbar = fig.colorbar(surf, shrink=0.6, aspect=6, pad=0.2)
    # cbar.ax.set_position(cbar.ax.get_position().translated(-0.05, 0))

    if metric == 'MCC':
        output_file_name = f'plot dataset {dataset}, {metric}, {classifier}'
        ts_writer.writerow(
            [classifier, dataset, round(max_mcc_z, 5), max_mcc_x, max_mcc_y, max_mcc_energy, round(threshold_z, 5),
             threshold_x, threshold_y, threshold_energy])
    else:
        # ax.zaxis.set_tick_params(pad=12)
        output_file_name = f'plot dataset {dataset}, energy consumption, {classifier}'

    plt.savefig(f'plots/pdf/{output_file_name}.pdf')

    plt.savefig(f'plots/png/{output_file_name}.png', dpi=300, transparent=True)

    readme_file.write(f'{classifier}\n')
    readme_file.write(f'![{classifier}](png/{quote(output_file_name)}.png)\n')

    plt.show()


classifier_list = ['RF', 'DT', 'AdaBoost', 'LogReg', 'Ridge']
dataset_list = ["IoTID20", "BoTNeTIoT-L01", "IoT-DNL", "X-IIoTID"]
metric_list = ['MCC', 'Energy consumption']

for dataset in dataset_list:
    readme_file.write(f'## Dataset {dataset}\n')
    for metric in metric_list:
        readme_file.write(f'### {metric}\n')
        for classifier in classifier_list:
            make_plot(dataset, classifier, metric)


# adding the efficiency plot to the Readme file.
readme_file.write(f'# Algorithm efficiency by dataset\n')
readme_file.write(f'![Algorithm efficiency by dataset](png/{quote("plot algorithm efficiency")}.png)\n')

# close the files
ts_file.close()
readme_file.close()

