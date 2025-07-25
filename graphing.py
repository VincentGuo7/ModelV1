import matplotlib.pyplot as plt
import numpy as np


# Example data
x = [1, 3, 5, 7]
# R2 = [0.8082, 0.6894, 0.6037, 0.5433]                # Line 2: y = x^2
# Z500 = [301.87, 406.87, 510.12, 603.69]                # Line 2: y = x^2
# T850 = [2.063, 2.785, 3.657, 4.048]                # Line 2: y = x^2
# T2M = [2.354, 3.083, 3.872, 4.008]                # Line 2: y = x^2
# U10 = [1.961, 2.579, 3.024, 3.375]           # Line 2: y = x^2
# V10 = [2.148, 2.899, 3.472, 3.916]                  # Line 2: y = x^2



# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, R2, label='R2', marker='s', color='red', markerfacecolor='red', markeredgecolor='red')
# plt.plot(x, T850, label='T850', marker='o', color='blue', markerfacecolor='blue', markeredgecolor='blue')
# plt.plot(x, T2M, label='T2M', marker='s', color='red', markerfacecolor='red', markeredgecolor='red')
# plt.plot(x, U10, label='U10', marker='^', color='green', markerfacecolor='green', markeredgecolor='green')
# plt.plot(x, V10, label='V10', marker='d', color='black', markerfacecolor='black', markeredgecolor='black')

# Add labels, legend, and title
plt.xlabel('Prediction Window (Days)', fontsize=16)
plt.xticks(x, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Model Performance, R2', fontsize=16)
# plt.title('Three Series on the Same Plot')
plt.legend(fontsize=14)
plt.grid(True)

# Save the figure
plt.savefig('V1R2.png', dpi=300, bbox_inches='tight')

# Optionally show the plot
plt.show()