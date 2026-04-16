import numpy as np
import matplotlib.pyplot as plt

# Want to plot one constant function and one linearly scaling function on the same graph, with the x-axis being m (the number of buses) and the y-axis being the time taken to solve the problem. The constant function should be a horizontal line at y=0.1 seconds, and the linear function should start at (0, 0) and increase to (100, 10) seconds.
m_values = np.linspace(0, 100, 100)
constant_time = 0.1 * np.ones_like(m_values)
linear_time = 0.003 * m_values
plt.figure(figsize=(10, 6))
plt.plot(m_values, constant_time, label='Distributed ALgorithm', color='blue')
plt.plot(m_values, linear_time, label='Newtons Method', color='orange')
plt.xlabel('Number of Buses (m)')
plt.ylabel('Time to Solve')
plt.title('Time to Solve vs Number of Buses')
plt.legend()
plt.grid(True)
plt.show()