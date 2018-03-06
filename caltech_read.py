from chainer.datasets import tuple_dataset
from PIL import Image
import numpy as np
import glob

pathsAndLabels = []
pathsAndLabels.append(np.asarray(["./imageDirectory0/", 0]))
pathsAndLabels.append(np.asarray(["./imageDirectory1/", 1]))
pathsAndLabels.append(np.asarray(["./imageDirectory2/", 2]))

# データを混ぜて、trainとtestがちゃんとまばらになるように。
allData = []
for pathAndLabel in pathsAndLabels:
    path = pathAndLabel[0]
    label = pathAndLabel[1]
    imagelist = glob.glob(path + "*")
    for imgName in imagelist:
        allData.append([imgName, label])
allData = np.random.permutation(allData)

imageData = []
labelData = []
for pathAndLabel in allData:
    img = Image.open(pathAndLabel[0])
    #3チャンネルの画像をr,g,bそれぞれの画像に分ける
    r,g,b = img.split()
    rImgData = np.asarray(np.float32(r)/255.0)
    gImgData = np.asarray(np.float32(g)/255.0)
    bImgData = np.asarray(np.float32(b)/255.0)
    imgData = np.asarray([rImgData, gImgData, bImgData])
    imageData.append(imgData)
    labelData.append(np.int32(pathAndLabel[1]))

threshold = np.int32(len(imageData)/8*7)
train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])