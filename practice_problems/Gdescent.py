import numpy as np

def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 100
    n = len(x)
    learning_rate = 0.08
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * np.sum((y - y_predicted) ** 2)
        md = -(2/n) * np.sum(x * (y - y_predicted))
        bd = -(2/n) * np.sum(y - y_predicted)
        m_curr -= learning_rate * md
        b_curr -= learning_rate * bd
        print(f"m={m_curr:.4f}, b={b_curr:.4f}, cost={cost:.4f}, iteration={i}")
    return m_curr, b_curr
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
m, b = gradient_descent(x, y)
print(f"\nFinal model: y = {m:.2f}x + {b:.2f}")
