import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import seaborn as sns

X1 = np.loadtxt("data/base_point_no_noise.txt")
X2 = np.loadtxt("data/base_point_noise.txt")
X3 = np.loadtxt("data/point_before_moving.txt")
X4 = np.loadtxt("data/point_after_moving.txt")

X1 = np.delete(X1, 112, axis = 1)
X2 = np.delete(X2, 112, axis = 1)
X3 = np.delete(X3, 112, axis = 1)
X4 = np.delete(X4, 112, axis = 1)

#X1 = np.concatenate([X1, np.ones((X1.shape[0], 1))], axis = 1)
#X2 = np.concatenate([X2, np.ones((X2.shape[0], 1))], axis = 1)
#print(X1.shape)
#print(X2.shape)
#print(X3.shape)
#print(X4.shape)

p = np.random.permutation(len(X1))
X1 = X1[p]
X2 = X2[p]

p = np.random.permutation(len(X3))
X3 = X3[p]
X4 = X4[p]

X1tr, X2tr, X3tr, X4tr = X1[:int(0.75 * len(X1))],X2[:int(0.75 * len(X2))], X3[:int(0.75 * len(X3))],X4[:int(0.75 * len(X4))]
X1te, X2te, X3te, X4te = X1[int(0.75 * len(X1)):],X2[int(0.75 * len(X2)):], X3[int(0.75 * len(X3)):],X4[int(0.75 * len(X4)):]

def shrink(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)
max_iter = 1000
lambda_ = 0.0005 #0.00025 -> great
w = np.ones(X1tr.shape[1])#np.random.normal(loc = 0, scale = 1, size = X1tr.shape[1])
w = w / np.linalg.norm(w)

X_1 = (X1tr - X2tr).T @ (X1tr - X2tr)
X_3 = (X3tr - X4tr).T @ (X3tr - X4tr)

alpha_1 = 0.1 / np.max(np.linalg.eig(X_1)[0]) #0.035 / len(X1tr) #0.035 -> great
alpha_3 = 0.1 / np.max(np.linalg.eig(X_3)[0])

loss = []
epsilon = 0.001
k = 3
W1 = np.random.normal(loc = 0, scale = 1, size = (X1.shape[1], k))
W1 = W1 / np.linalg.norm(W1)
W2 = np.random.normal(loc = 0, scale = 1, size = (k,))
W2 = W2 / np.linalg.norm(W2)
w1prev = np.zeros(W1.shape)
w2prev = np.zeros(W2.shape)
for i in range(max_iter):
    linear1_move1 = X1 @ W1
    linear1_move2 = X2 @ W1
    relu_move1 = np.maximum(linear1_move1, 0)
    relu_move2 = np.maximum(linear1_move2, 0)
    linear2_move1 = relu_move1 @ W2
    linear2_move2 = relu_move2 @ W2

    linear1_noise1 = X1 @ W1
    linear1_noise2 = X2 @ W1
    relu_noise1 = np.maximum(linear1_noise1, 0)
    relu_noise2 = np.maximum(linear1_noise2, 0)
    linear2_noise1 = relu_noise1 @ W2
    linear2_noise2 = relu_noise2 @ W2

    mask1 = relu_move1[relu_move1 > 0] = 1
    mask2 = relu_move2[relu_move2 > 0] = 1
    mask3 = relu_noise1[relu_noise1 > 0] = 1
    mask4 = relu_noise2[relu_noise2 > 0] = 1

    W2_grad = alpha_1 * relu_move1.T @ relu_move1 @ W2 + alpha_3 * relu_noise1.T @ relu_noise1 @ W2

    W1_grad = alpha_1 * np.sum(X1[:, :, np.newaxis] @ (W2 * mask1).reshape((1, k)), axis = 0) - alpha_1 * np.sum(X2[:, :, np.newaxis] @ (W2 * mask2).reshape((1, k)), axis = 0) + alpha_3 * np.sum(X3[:, :, np.newaxis] @ (W2 * mask3).reshape((1, k)), axis = 0)+ alpha_3 * np.sum(X4[:, :, np.newaxis] @ (W2 * mask4).reshape((1, k)), axis = 0)

    W1 = shrink(W1 - W1_grad, lambda_)
    W1 = W1 / np.linalg.norm(W1)

    W2 = shrink(W2 - W2_grad, lambda_)
    W2 = W2 / np.linalg.norm(W2)
    if (np.linalg.norm(W1 - w1prev) < epsilon and np.linalg.norm(W2 - w2prev) < epsilon):
        break
    w1prev = W1.copy()
    w2prev = W2.copy()
    loss.append(lambda_ * np.linalg.norm(W1, ord = 1) + lambda_ * np.linalg.norm(W2, ord = 1) + (1 / len(linear2_move1)) * np.linalg.norm(linear2_move1 - linear2_move2) + (1 / len(linear2_noise1)) * np.linalg.norm(linear2_noise1 - linear2_noise2))
plt.plot(loss)
plt.show()
print("sparsity:", len(np.where(w > epsilon)[0]))

print


img_id = 80
#U, S, Vt = np.linalg.svd(X)
#w = U[:,-1]
X = np.loadtxt('/Users/farhantoddywala/desktop/dataset2/poses/04.txt')
X = X.reshape((X.shape[0], 3, 4))

img1 = cv2.imread('/Users/farhantoddywala/desktop/dataset/sequences/04/image_0/'+str(img_id).zfill(6)+'.png', 0)
img2 = cv2.imread('/Users/farhantoddywala/desktop/dataset/sequences/04/image_0/'+str(img_id + 1).zfill(6)+'.png', 0)
R1 = X[img_id][:,:3]
R2 = X[img_id + 1][:,:3]
K = np.loadtxt('/Users/farhantoddywala/desktop/dataset/sequences/04/calib.txt')

W1 = np.insert(w, 112, 0, axis = 0)
def corresponding(x1, R1, R2, K):
    return K @ R2 @ np.linalg.pinv(R1) @ np.linalg.pinv(K) @ x1

def kp_to_homog(kp):
    s = set([])
    arr = []
    for marker in kp:
        a, b =  tuple(int(i) for i in marker.pt)
        s.add((a, b))
        arr.append([a, b, 1])
    return s
fast = cv2.FastFeatureDetector_create(25)

offset = 7
interesting = np.zeros(img1.shape)
values = []
for i in range(offset, img1.shape[0] - offset):
    for j in range(offset, img1.shape[1] - offset):
        tmp = (img1[i - offset : i + offset + 1, j - offset : j + offset + 1] - img1[i][j]).flatten() / 255
        interesting[i][j] = W2 @ np.maximum(W1.T @ tmp, 0)
        values.append(interesting[i][j])

thres = sorted(values)[int(0.995 * len(values))]
interesting[interesting < thres] = 0
interesting2 = np.zeros(img1.shape)
cnt = 0
for i in range(offset, img1.shape[0] - offset):
    for j in range(offset, img1.shape[1] - offset):
        tmp = (img2[i - offset : i + offset + 1, j - offset : j + offset + 1] - img2[i][j]).flatten() / 255
        interesting2[i][j] = W2 @ np.maximum(W1.T @ tmp, 0)
        if (interesting2[i][j] >= thres):
            cnt += 1

interesting2[interesting2 < thres] = 0
#plt.imshow(interesting, cmap = 'gray')
#plt.show()
#plt.imshow(interesting2, cmap = 'gray')
#plt.show()

coordinates1_true = []
coordinates_pred = []
corr = []
corrkp = []
kp1 = fast.detect(img1, None)
kp2 = fast.detect(img2, None)
MSE = 0
s1 = kp_to_homog(kp1)
s2 = kp_to_homog(kp2)
window = 4
for i in range(offset, img1.shape[0] - offset):
    for j in range(offset, img1.shape[1] - offset):
        if ((i, j) in s1):
            c2 = corresponding(np.array([i, j, 1]), R1, R2, K)
            c2 = (c2 / c2[2]).astype("int")
            worked2 = 0
            for n in range(-window, window + 1):
                for m in range(-window, window + 1):
                    if ((c2[0] - n, c2[1] - m) in s2):
                        worked2 = 1
            corrkp.append(worked2)
        if (interesting[i][j] >= thres):
            coor = np.array([i, j, 1])
            coordinates1_true.append(coor)
            coor2 = corresponding(coor, R1, R2, K)
            coor2 = (coor2 / coor2[2]).astype("int")
            worked = 0
            for n in range(-window, window + 1):
                for m in range(-window, window + 1):
                    if (interesting2[coor2[0] + n][coor2[1] + m] >= thres):
                        worked = 1
                    if (n == 0 and m == 0):
                        MSE += (interesting[i][j] - interesting2[i][j]) ** 2
            corr.append(worked)
        if (interesting2[i][j] >= thres):
            coor = np.array([i, j, 1])
            coordinates_pred.append(coor)

MSE = MSE / len(coordinates1_true)
print("MSE no noise:", MSE)
print("number of interesting points in image 1:", int(0.005 * len(values)))
print("number of interesting points in image 2:", cnt)
print("percentage of interesting points moved: " + str(100 * sum(corr)/len(corr)) + "%")
coordinates1_true = np.array(coordinates1_true)
corr = np.array(corr)
coordinates_pred = np.array(coordinates_pred)
print("number of corners in image 1:", len(kp1))
print("number of corners image 2:", len(kp2))
print("percentage of corners moved: " + str(100 * sum(corrkp)/len(corrkp)) + "%")
print()
'''
plt.imshow(interesting, cmap = "hot")
plt.show()
plt.imshow(interesting2, cmap = "hot")
plt.show()
'''
sns.heatmap(interesting)
sns.heatmap(interesting2)
def test_between_images_with_noise(img1, img2, R1, R2, K, noise, thres, w, old1, old2):
    img1 = img1 + np.random.normal(loc = 0, scale = noise, size = img1.shape)
    img2 = img2 + np.random.normal(loc = 0, scale = noise, size = img1.shape)
    img1[img1 > 255] = 255
    img1[img1 < 0] = 0
    img1 = img1.astype("uint8")
    img2[img2 > 255] = 255
    img2[img2 < 0] = 0
    img2 = img2.astype("uint8")
    offset = 7
    interesting = np.zeros(img1.shape)
    cnt2 = 0
    for i in range(offset, img1.shape[0] - offset):
        for j in range(offset, img1.shape[1] - offset):
            tmp = (img1[i - offset : i + offset + 1, j - offset : j + offset + 1] - img1[i][j]).flatten() / 255
            interesting[i][j] = W2 @ np.maximum(W1.T @ tmp, 0)
            if (interesting[i][j] >= thres):
                cnt2 += 1

    interesting[interesting < thres] = 0

    interesting2 = np.zeros(img1.shape)
    cnt = 0
    for i in range(offset, img1.shape[0] - offset):
        for j in range(offset, img1.shape[1] - offset):
            tmp = (img2[i - offset : i + offset + 1, j - offset : j + offset + 1] - img2[i][j]).flatten() / 255
            interesting2[i][j] = W2 @ np.maximum(W1.T @ tmp, 0)
            if (interesting2[i][j] >= thres):
                cnt += 1

    interesting2[interesting2 < thres] = 0
    #plt.imshow(interesting, cmap = 'gray')
    #plt.show()
    #plt.imshow(interesting2, cmap = 'gray')
    #plt.show()

    coordinates1_true = []
    coordinates_pred = []
    corr = []
    corrkp = []
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    MSE = 0
    s1 = kp_to_homog(kp1)
    s2 = kp_to_homog(kp2)
    for i in range(offset, img1.shape[0] - offset):
        for j in range(offset, img1.shape[1] - offset):
            if ((i, j) in s1):
                c2 = corresponding(np.array([i, j, 1]), R1, R2, K)
                c2 = (c2 / c2[2]).astype("int")
                worked2 = 0
                for n in range(-window, window + 1):
                    for m in range(-window, window + 1):
                        if ((c2[0] - n, c2[1] - m) in s2):
                            worked2 = 1
                corrkp.append(worked2)
            if (interesting[i][j] >= thres):
                coor = np.array([i, j, 1])
                coordinates1_true.append(coor)
                coor2 = corresponding(coor, R1, R2, K)
                coor2 = (coor2 / coor2[2]).astype("int")
                worked = 0
                for n in range(-window, window + 1):
                    for m in range(-window, window + 1):
                        if (interesting2[coor2[0] + n][coor2[1] + m] >= thres):
                               worked = 1
                        if (n == 0 and m == 0):
                            MSE += (interesting[i][j] - interesting2[i][j]) ** 2
                corr.append(worked)
            if (interesting2[i][j] >= thres):
                coor = np.array([i, j, 1])
                coordinates_pred.append(coor)

    MSE = MSE / len(coordinates1_true)
    print("MSE noise", noise, ":", MSE)
    print("number of interesting points in image 1:", cnt2)
    print("number of interesting points in image 2:", cnt)
    print("percentage of interesting points moved: " + str(100 * sum(corr)/len(corr)) + "%")

    coordinates1_true = np.array(coordinates1_true)
    corr = np.array(corr)
    coordinates_pred = np.array(coordinates_pred)
    print("number of corners in image 1:", len(kp1))
    print("number of corners image 2:", len(kp2))
    print("percentage of corners moved: " + str(100 * sum(corrkp)/len(corrkp)) + "%")
    interesting[old1 == 0] = 0
    interesting2[old2 == 0] = 0
    print("MSE vs non-noisy version of image 1:", (np.linalg.norm(interesting - old1) ** 2) / np.count_nonzero(interesting))
    print("MSE vs non-noisy version of image 2:", (np.linalg.norm(interesting2 - old2) ** 2) / np.count_nonzero(interesting2))
    print()

for noise in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
    test_between_images_with_noise(img1, img2, R1, R2, K, noise, thres, w, interesting, interesting2)
