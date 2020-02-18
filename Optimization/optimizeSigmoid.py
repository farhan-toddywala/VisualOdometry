import numpy as np


X1 = np.loadtxt("X1.txt") / 255
X2 = np.loadtxt("X2.txt") / 255
X1 = np.concatenate([X1, np.ones((X1.shape[0], 1))], axis = 1)
X2 = np.concatenate([X2, np.ones((X2.shape[0], 1))], axis = 1)

max_iter = 1000
w = np.zeros(X1.shape[1])
alpha = 0.000001
k = 0.001 * len(X1)
b = 1 / len(X1) #0.01 worked
epsilon = 0.0001
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

loss = float("inf")
for i in range(max_iter):
    
    print("iteration:", i)
    tmp = X1 - X2
    linear = X1 @ w
    sig = sigmoid(linear)
    sum_ = np.sum(sig)
    quad = b * (k - sum_) ** 2
    quad_loss = -b * (k - sum_)
    sum_loss = quad_loss * np.ones(sig.shape)
    sig_loss = sigmoid(sum_loss) * (1 - sigmoid(sum_loss))
    linear_loss = X1.T @ sig_loss
    prevloss = loss
    loss = (1 / len(X1)) * np.linalg.norm(X1 @ w - X2 @ w) ** 2 + quad
    wprev = w.copy()
    w = w - alpha * ((1 / len(X1)) * tmp.T @ tmp @ w + linear_loss)
    #print(quad_loss)
    #print(sum_loss)
    #print(sig_loss)
    #print(linear_loss)
    #print()
    if (np.linalg.norm(w - wprev) < epsilon or prevloss < loss):
        break

print(w)
print(np.sum(sigmoid(X1 @ w)))
print(np.linalg.norm(X1 @ w - X2 @ w) ** 2)
    
