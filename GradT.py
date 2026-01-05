import numpy as np
import matplotlib.pyplot as plt
def gradient_descent(x, y):
    m_curr = 0
    b_curr = 0
    rate = 0.01
    n = len(x)
    for i in range(10000):
        y_predicted = m_curr * x + b_curr
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr -= rate * md
        b_curr -= rate * bd
    return m_curr, b_curr
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
m, b = gradient_descent(x, y)
y_predicted = m * x + b
plt.scatter(x, y, color='blue', marker='+')
plt.plot(x, y_predicted, color='red')
plt.show()