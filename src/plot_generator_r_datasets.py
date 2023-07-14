import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# running device MIPS
real_mips = 63250

# simulated device specs
# HP ProLiant ML110 G4, Intel Xeon 3075  / 4GB / 64GB (MIPS: 5320, Power: 93.7 - 135)
# ref: Khan, A. A., Zakarya, M., & Khan, R. (2019). Energy-aware dynamic resource management in elastic cloud datacenters. Simulation modelling practice and theory, 92, 82-99

simulated_mips = 5320
simulated_idle_energy = 93.7
simulated_busy_energy = 135

aux_constant = (simulated_busy_energy - simulated_idle_energy) / 100


df = pd.read_csv('results_by_dataset.csv')

df['Training time'] = df['Training time'] * (real_mips / simulated_mips)
df['Training Energy consumption'] = df['Training time'] * (simulated_idle_energy + aux_constant * df['TR-CPU%'])

df['Testing time'] = df['Testing time'] * (real_mips / simulated_mips)
df['Testing Energy consumption'] = df['Testing time'] * (simulated_idle_energy + aux_constant * df['TE-CPU%'])

df['Energy consumption by second'] = df['Training time'] / df['Training Energy consumption']

# Filter the data for the desired classifier
# sns.set_palette("Set1")

plt.figure(figsize=(8, 6))

sns.barplot(data=df, x='DATASET', y='Energy consumption by second', hue='Classifier',)

# plt.xlabel('Classifier')
# plt.ylabel('Accuracy')

plt.xticks(rotation=45)

plt.legend()

plt.subplots_adjust(right=0.5)  # Adjust the right margin here

plt.tight_layout()  # Ensures the labels and ticks are not cut off

plt.legend(loc='lower right')

plt.show()
