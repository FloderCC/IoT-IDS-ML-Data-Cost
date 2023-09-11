import seaborn as sns
import matplotlib.pyplot as plt

# Sample data for the heatmap
data = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]

# Create a heatmap using the RdYlBu colormap
sns.heatmap(data, cmap="RdYlGn_r")

# Display the heatmap
plt.show()
