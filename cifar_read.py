import chainer
from chainer import cuda
import dataset as D

if __name__ == '__main__':
    data = D.Dataset('cifar10',valid_data_ratio=0.)
    x = chainer.Variable(cuda.to_gpu(data.x_train))
    t = chainer.Variable(cuda.to_gpu(data.y_train))

    print(x)