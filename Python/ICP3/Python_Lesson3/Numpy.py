import numpy as np

x = np.random.randint(1, 20, size=15)
print("Original vector:")
print(x)
x[np.where(x == np.max(x))] = 0
print("Replacing maximum value by 0")
print(x)
