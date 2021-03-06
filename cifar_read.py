import chainer
import dataset as D
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    data = D.Dataset('mit',valid_data_ratio=0.)
    x = chainer.Variable(data.x_train)
    t = chainer.Variable(data.y_train)


    kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9]])

    dst = cv2.filter2D(data.x_train[3][0],-1,kernel)
    cv2.imwrite("test_before.png",data.x_train[3][0])
    #cv2.imwrite("test_before2.png", x[1][0])
    cv2.imwrite("test.png",dst)

    '''
    data.x_train[A][B]
    Aには画像の種類
    Bにはチャネル(0~2)
    '''