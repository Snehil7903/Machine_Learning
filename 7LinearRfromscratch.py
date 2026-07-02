import numpy as np

X = np.array([1,2,3,4,5])
y = np.array([35,40,50,60,70])

w = 0
b = 0

learning_rate = 0.01
epochs = 1000

n = len(X)

for epoch in range(epochs):

    y_pred = w * X + b

    dw = (2/n) * np.sum(X * (y_pred - y))
    db = (2/n) * np.sum(y_pred - y)

    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Weight:", w)
print("Bias:", b)

print(y_pred)