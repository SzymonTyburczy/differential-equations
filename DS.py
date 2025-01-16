import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MES import ei


df = pd.read_csv('results.csv')
u_values = df['ui']

n = len(u_values)
h = 2.0 / n
x_values = np.linspace(0.0, 2.0, 1000)
y_values = []

for i in range(1000):
    f_value = 0.0
    for j in range(n):
        f_value += u_values[j] * ei(j * h, h, x_values[i])
    y_values.append(f_value)

plt.plot(x_values, y_values)
plt.grid()
plt.title('Równanie transportu ciepła, n = ' + str(n))
plt.xlabel('Wartości argumentów x funkcji u(x)')
plt.ylabel('Wartości poszukiwanej funkcji u(x)')
plt.show()
