import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import seaborn as sns

Noisy = np.loadtxt("NoisyMove.txt")
NonNoisy = np.loadtxt("NonNoisyMove.txt")
Between = np.loadtxt("NonToNoisy.txt")
Order1 = np.loadtxt("Order1.txt")
Order2 = np.loadtxt("Order2.txt")
Noisy = np.delete(Noisy, 112, axis = 1)
NonNoisy = np.delete(NonNoisy, 112, axis = 1)
Order1 = np.delete(Order1, 112, axis = 1)
Order2 = np.delete(Order2, 112, axis = 1)
Between = np.delete(Between, 112, axis = 1)

X1 = Noisy
X2 = NonNoisy
X3 = Between
X4 = Order1
X5 = Order2
#X1 = np.concatenate([X1, np.ones((X1.shape[0], 1))], axis = 1)
#X2 = np.concatenate([X2, np.ones((X2.shape[0], 1))], axis = 1)
print(X1.shape)
print(X2.shape)
print(X3.shape)
print(X4.shape)
print(X5.shape)
np.random.shuffle(X1)
np.random.shuffle(X2)
np.random.shuffle(X3)
p = np.random.permutation(len(X4))
X4 = X4[p]
X5 = X5[p]

X1tr, X2tr, X3tr, X4tr, X5tr = X1[:int(0.75 * len(X1))],X2[:int(0.75 * len(X2))], X3[:int(0.75 * len(X3))],X4[:int(0.75 * len(X4))], X5[:int(0.75 * len(X5))]
X1te, X2te, X3te, X4te, X5te = X1[int(0.75 * len(X1)):],X2[int(0.75 * len(X2)):], X3[int(0.75 * len(X3)):],X4[int(0.75 * len(X4)):], X5[int(0.75 * len(X5)):]

def shrink(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)
max_iter = 1000
lambda_ = 0.0004 #0.00025 -> great
w = np.ones(X1tr.shape[1])#np.random.normal(loc = 0, scale = 1, size = X1tr.shape[1])
w = w / np.linalg.norm(w)

X_1 = X1tr.T @ X1tr
X_2 = X2tr.T @ X2tr
X_3 = X3tr.T @ X3tr
X_4 = X4tr.T @ X5tr + X5tr.T @ X4tr
alpha_1 = 0.1 / np.max(np.linalg.eig(X_1)[0]) #0.035 / len(X1tr) #0.035 -> great
alpha_2 = 0.1 / np.max(np.linalg.eig(X_2)[0])
alpha_3 = 0.1 / np.max(np.linalg.eig(X_3)[0])
alpha_4 = 0.0 / len(X4tr)
loss = []
epsilon = 0.001

for i in range(max_iter):
    wprev = w.copy()
    u1 = X4tr @ w
    u2 = X5tr @ w
    #print(u1.shape)
    #print(u2.shape)
    v1 = u1 * u2
    #print(v1.shape)
    x1 = np.maximum(-v1, 0)
    #print(x1.shape)
    z1 = x1.copy()
    z1[z1 > 0] = 1
    grad = X5tr.T @ (X4tr @ w * z1) + X4tr.T @ (X5tr @ w * z1)
    w = shrink(w - alpha_1 * X_1 @ w - alpha_2 * X_2 @ w - alpha_3 * X_3 @ w - alpha_4 * grad, lambda_)
    w = w / np.linalg.norm(w)
    if (np.linalg.norm(w - wprev) < epsilon):
        break
    loss.append(np.linalg.norm(X1tr @ w) ** 2 + np.linalg.norm(X2tr @ w) ** 2 + np.linalg.norm(X3tr @ w) ** 2 + np.sum(x1))
plt.plot(loss)
plt.show()
print("sparsity:", len(np.where(w > epsilon)[0]))



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

w = np.insert(w, 112, 0)
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
        interesting[i][j] = tmp.T @ w
        values.append(interesting[i][j])

thres = sorted(values)[int(0.995 * len(values))]
interesting[interesting < thres] = 0
interesting2 = np.zeros(img1.shape)
cnt = 0
for i in range(offset, img1.shape[0] - offset):
    for j in range(offset, img1.shape[1] - offset):
        tmp = (img2[i - offset : i + offset + 1, j - offset : j + offset + 1] - img2[i][j]).flatten() / 255
        interesting2[i][j] = tmp.T @ w
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
            interesting[i][j] = tmp.T @ w
            if (interesting[i][j] >= thres):
                cnt2 += 1

    interesting[interesting < thres] = 0

    interesting2 = np.zeros(img1.shape)
    cnt = 0
    for i in range(offset, img1.shape[0] - offset):
        for j in range(offset, img1.shape[1] - offset):
            tmp = (img2[i - offset : i + offset + 1, j - offset : j + offset + 1] - img2[i][j]).flatten() / 255
            interesting2[i][j] = tmp.T @ w
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
