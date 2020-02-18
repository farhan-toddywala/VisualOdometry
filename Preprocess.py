import numpy as np
import cv2

sequence = "03"

filter = np.loadtxt("/Users/farhantoddywala/desktop/research/filter4.txt").reshape((15, 15))
print(np.count_nonzero(filter))
offset = (len(filter) - 1) // 2
def get_interest_points(img, thres, label):
    folder2 = '/Users/farhantoddywala/desktop/preprocessed_imgs/' + sequence + "-" + str(noise_std) + '/'
    H = img.shape[0]
    W = img.shape[1]
    interesting_points = []
    non_max = np.zeros(img.shape)
    for i in range(len(filter) + 1, H - len(filter) - 1):
        for j in range(len(filter) + 1, H - len(filter) - 1):
            tmp = (img[i - offset : i + offset + 1, j - offset : j + offset + 1] - img[i][j]).flatten() / 255
            score = tmp.T @ filter.flatten()
            non_max[i][j] = score
            if (score >= thres and score > non_max[i - 1][j] and score > non_max[i - 1][j - 1] and score > non_max[i][j - 1]):
                interesting_points.append([i,j])
    interest_pts = np.array(interesting_points)
    print("number of interest points in image " + str(label) + ":", interest_pts.shape[0])
    np.savetxt(folder2 + str(label) + "_interest_points", interest_pts)

num_imgs = 802
thres = 1.05#1.
noise_std = 40
folder = '/Users/farhantoddywala/desktop/prenoise/'

for i in range(num_imgs - 2, num_imgs):
    name = folder + "0" * (6 - len(str(i))) + str(i)
    img = cv2.imread(folder+str(i)+'.png', 0)
    #img = img + np.random.normal(loc = 0, scale = noise_std, size = img.shape)
    #img[img > 255] = 255
    #img[img < 0] = 0
    #img = img.astype("uint8")
    if (img is None):
        print("image " + str(i) + " not found")
        continue
    if (i == 0):
        print("number of pixels in image:", img.shape[0] * img.shape[1])
    get_interest_points(img, thres, i)
