import numpy as np
import cv2

from visual_odometry import PinholeCamera, VisualOdometry

#noise_std = 60
high = 2000
low = 1000
increase_rate = 1.1
decrease_rate = 0.9
print("upper level of interest points allowed:", high)
print("lower level of interest points allowed:", low)
num_trials = 9
K = np.loadtxt('/Users/farhantoddywala/desktop/dataset/sequences/06/calib.txt')
X = np.loadtxt('/Users/farhantoddywala/desktop/dataset2/poses/06.txt')
print(X.shape)
X = X.reshape((X.shape[0], 3, 4))
#dynamic = 1
noise_std = 0

for noise_std in [5, 15, 25, 35, 45, 55]:
    for dynamic in [0, 1]:
        name = "dynamic"
        #print("upper limit of noise std:", upper)
        print("noise std:", noise_std)
        if (dynamic == 0):
            print('fixed threshold')
            name = "fixed"
        else:
            print("dynamic_threshold")
        mse_lst = []
        for k in range(num_trials):
            cam = PinholeCamera(1226.0, 370.0, 718.8560, 718.8560, 607.1928, 185.2157)
            vo = VisualOdometry(cam, '/Users/farhantoddywala/desktop/dataset/sequences/06/calib2.txt')
            traj = np.zeros((600,600,3), dtype=np.uint8)
            scale = 1.0
            predicted = []
            true = []
            #noise_std = 0
            for img_id in range(0, len(X)):
                    img = cv2.imread('/Users/farhantoddywala/desktop/dataset/sequences/06/image_0/'+str(img_id).zfill(6)+'.png', 0)
                    #img = cv2.resize(img, (620,188))
                    #print(img[0][0])
                    #print(img.shape)
                    #rand = np.random.randint(low = -1, high = 2)
                    #noise_std = min(max(noise_std + rand, 0), upper)
                    img = img + np.random.normal(loc = 0, scale = noise_std, size = img.shape)
                    img[img > 255] = 255
                    img[img < 0] = 0
                    img = img.astype("uint8")
                    #print(img.shape)
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if (img_id > 0):
                            scale = np.linalg.norm(X[img_id][:,3] - X[img_id - 1][:,3])
                    vo.scale = scale
                    vo.update(img, img_id)
                    cur_t = vo.cur_t
                    kp = len(vo.detector.detect(img))
                    #print(kp)
                    if (dynamic == 1):
                        if (kp > high):
                                vo.detector = cv2.FastFeatureDetector_create((int)(vo.detector.getThreshold() * increase_rate), nonmaxSuppression=True)
                        if (kp < low):
                                vo.detector = cv2.FastFeatureDetector_create((int)(vo.detector.getThreshold() * decrease_rate), nonmaxSuppression=True)
                    if(img_id > 0):
                            x, y, z = cur_t[0], cur_t[1], cur_t[2]
                    else:
                            x, y, z = 0., 0., 0.
                    draw_x, draw_y = int(x) // 5 +180, int(z)//5 +270
                    true_x, true_y = (int)(X[img_id][:,3][0]) // 5 + 180, (int)(X[img_id][:,3][2]) // 5 + 270
                    #print(draw_x, draw_y)
                    #print(true_x, true_y)
                    #cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
                    #cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
                    #cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
                    #text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
                    #cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
                    #cv2.imshow('Road facing camera', img)
                    #if (img_id >= 0):
                    #cv2.imshow('Trajectory', traj)
                    #cv2.waitKey(10)
                    predicted.append([x, z])
                    true.append([X[img_id][:,3][0], X[img_id][:,3][2]])

            #cv2.waitKey(5)
            predicted = np.array(predicted)
            true = np.array(true)
            print("MSE:",(np.linalg.norm(true - predicted) ** 2) / len(true))
            mse_lst.append((np.linalg.norm(true - predicted) ** 2) / len(true))


            cv2.imwrite('map.png', traj)
        mse_lst = np.array(mse_lst)
        print(mse_lst)
        np.savetxt(str(noise_std) + "_" + name + "_noise_seq6.txt", mse_lst)
