"""File description:

Script for generating bar plot about energy efficiency.

Note that the variable 'real_mips' should represent the number of MIPS available on the computer where the experiment
was executed.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# running device MIPS
real_mips = 63250

# simulated device specs
# HP ProLiant ML110 G4, Intel Xeon 3075  / 4GB / 64GB (MIPS: 5320, Power: 93.7 - 135)
# ref: Khan, A. A., Zakarya, M., & Khan, R. (2019). Energy-aware dynamic resource management in elastic cloud datacenters. Simulation modelling practice and theory, 92, 82-99

simulated_mips = 5320
simulated_idle_energy = 93.7
simulated_busy_energy = 135

aux_constant = (simulated_busy_energy - simulated_idle_energy) / 100

df = pd.read_csv(f'results/results_by_rows_and_features.csv')

# filtering the results by sampling seeds
df = df[df['Random seed'] == 'AVG']

# computing and adding the energy consumption
df['Training time'] = df['Training time'] * (real_mips / simulated_mips)
df['Energy consumption'] = df['Training time'] * (simulated_idle_energy + aux_constant * df['TR-CPU%'])

# computing the ratio column
df['MCC / J'] = df['MCC'] / df['Energy consumption']

# Define the custom order for "Dataset name"
dataset_order = ["IoTID20", "BoTNeTIoT-L01", "X-IIoTID", "IoT-DNL"]  # Add the dataset names in the desired order

# Create the bar plot with specified order for "Dataset name"
sns.barplot(data=df, x="Classifier", y="MCC / J", hue="Dataset name", hue_order=dataset_order)

# # Add y-values to the bars
# for p in plt.gca().patches:
#     plt.gca().annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')


# Save the plot to a PDF file
plt.savefig('plots/pdf/plot algorithm efficiency.pdf')
plt.savefig('plots/png/plot algorithm efficiency.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

