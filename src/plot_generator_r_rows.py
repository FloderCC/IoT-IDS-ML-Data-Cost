import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

# running device MIPS
real_mips = 63250

# simulated device specs
# HP ProLiant ML110 G4, Intel Xeon 3075  / 4GB / 64GB (MIPS: 5320, Power: 93.7 - 135)
# ref: Khan, A. A., Zakarya, M., & Khan, R. (2019). Energy-aware dynamic resource management in elastic cloud datacenters. Simulation modelling practice and theory, 92, 82-99

simulated_mips = 5320
simulated_idle_energy = 93.7
simulated_busy_energy = 135

aux_constant = (simulated_busy_energy - simulated_idle_energy) / 100

df = pd.read_csv('results_by_rows.csv')

# max_time = df['Training time'].max()

df['Training time'] = df['Training time'] * (real_mips / simulated_mips)
df['Energy consumption'] = df['Training time'] * (simulated_idle_energy + aux_constant * df['TR-CPU%'])

df = df[df['Classifier'] == 'KNN']

host = host_subplot(111, axes_class=axisartist.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))

par1.axis["right"].toggle(all=True)
par2.axis["right"].toggle(all=True)


p1, = host.plot(df['Dataset %'], df['Accuracy'], label="Accuracy")
p4, = host.plot(df['Dataset %'], df['f1-score'], label="f1-score")
p5, = host.plot(df['Dataset %'], df['MCC'], label="MCC")


p2, = par1.plot(df['Dataset %'], df['Training time'], label="Training time")
p3, = par2.plot(df['Dataset %'], df['Energy consumption'], label="Energy consumption")

host.set(xlabel="Dataset %", ylabel="Value")
par1.set(ylabel="Training time")
par2.set(ylabel="Energy consumption")

host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())

plt.show()