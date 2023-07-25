"""File description:

Script for generating the spearman coefficients summary.

Note that the variable 'real_mips' should represent the number of MIPS available on the computer where the experiment
was executed.
"""

import csv

import pandas as pd
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

def get_coefficient(classifier, dataset, col1, col2):
    # loading the results
    df = pd.read_csv(f'results/results_by_rows_and_features.csv')

    # filtering the results by dataset & classifier & AVG of sampling seeds
    df = df[df['Dataset name'] == dataset]
    df = df[df['Classifier'] == classifier]
    df = df[df['Random seed'] == 'AVG']

    df['Training time'] = df['Training time'] * (real_mips / simulated_mips)
    df['Energy consumption'] = df['Training time'] * (simulated_idle_energy + aux_constant * df['TR-CPU%'])

    # correlation = df[col1].corr(df[col2])
    correlation, p_value = spearmanr(df[col1], df[col2])
    print(f"Spearman's correlation coefficient: {correlation}")
    print(f"P-value: {p_value}")

    return str(correlation)
    # return str(correlation) + ' (' + str(p_value) + ')'


f = open('results/spearman_coefficients.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(['Classifier', 'Dataset', 'Instances VS energy SRCC', 'Features VS energy SRCC', 'Instances VS MCC SRCC', 'Features VS MCC SRCC', 'MCC VS energy SRCC'])


classifier_list = ['RF', 'DT', 'AdaBoost', 'LogReg', 'Ridge']
dataset_list = ["IoTID20", "BoTNeTIoT-L01", "IoT-DNL", "X-IIoTID"]
metric_list = ['MCC', 'Energy consumption']

results = []
for classifier in classifier_list:
    for dataset in dataset_list:
        results.append([classifier, dataset,
                         get_coefficient(classifier, dataset, 'Dataset %', 'Energy consumption'),
                         get_coefficient(classifier, dataset, 'Number of features', 'Energy consumption'),
                         get_coefficient(classifier, dataset, 'Dataset %', 'MCC'),
                         get_coefficient(classifier, dataset, 'Number of features', 'MCC'),
                         get_coefficient(classifier, dataset, 'MCC', 'Energy consumption')
                         ])

# Calculate the averages for each column excluding the first two columns
num_columns = len(results[0])  # Number of columns in the table
num_rows = len(results)        # Number of rows in the table

averages = [sum(float(results[i][j]) for i in range(num_rows)) / num_rows for j in range(2, num_columns)]

# Create a new row with the calculated averages
new_row = ['AVG', 'AVG'] + [str(average) for average in averages]

results.append(new_row)

writer.writerows(results)


# close the file
f.close()

