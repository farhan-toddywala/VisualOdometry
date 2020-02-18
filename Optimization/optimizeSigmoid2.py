import numpy as np


X1 = np.loadtxt("X1.txt") / 255
X2 = np.loadtxt("X2.txt") / 255
X1 = np.concatenate([X1, np.ones((X1.shape[0], 1))], axis = 1)
X2 = np.concatenate([X2, np.ones((X2.shape[0], 1))], axis = 1)

max_iter = 1000
w = np.random.normal(loc = 0, scale = 1 / X1.shape[1] ** 2,size = X1.shape[1])
alpha = 0.001
k = 1
b = 0.001#0.01 worked
epsilon = 0.00000001
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

loss = float("inf")
for i in range(max_iter):
    
    print("iteration:", i)
    tmp = X1 - X2
    linear = X1 @ w
    sum_ = np.linalg.norm(linear) ** 2
    quad = b * (k - sum_) ** 2
    quad_loss = -b * (k - sum_)
    norm_loss = quad_loss * linear
    linear_loss = X1.T @ norm_loss
    prevloss = loss
    same_loss = (1 / len(X1)) * np.linalg.norm(X1 @ w - X2 @ w) ** 2
    loss = same_loss + quad
    wprev = w.copy()
    w = w - alpha * ((1 / len(X1)) * tmp.T @ tmp @ w + linear_loss)
    print(np.max(linear_loss))
    print(np.max((1 / len(X1)) * tmp.T @ tmp @ w))
    #print(sum_loss)
    #print(sig_loss)
    #print(linear_loss)
    #print()
    if (np.linalg.norm(w - wprev) < epsilon):
        break

print(w)
print(np.linalg.norm(X1 @ w) ** 2)
print(np.linalg.norm(X1 @ w - X2 @ w) ** 2)
    
