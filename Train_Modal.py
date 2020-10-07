import numpy as np
from Model_arch import alexnet
import random
import glob
import cv2

WIDTH = 256
HEIGHT = 256
LR = 1e-3
EPOCHS = 30
MODEL_NAME = 'D:/Projects/Emotion_detection/models/no_{}name-{}-epochs-{}.model'.format(LR, 'final_3',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)
#model.load('D:/Projects/Emotion_detection/models/no_{}name-{}-epochs-{}.model'.format(LR, 'final_3',EPOCHS))

temp1 = glob.glob("D:/Projects/Emotion_detection/data_set/angry/*.JPG")
temp2 = glob.glob("D:/Projects/Emotion_detection/data_set/happy/*.JPG")
temp3 = glob.glob("D:/Projects/Emotion_detection/data_set/sad/*.JPG")
temp4 = glob.glob("D:/Projects/Emotion_detection/data_set/surprised/*.JPG")
temp5 = glob.glob("D:/Projects/Emotion_detection/data_set/unknown/*.JPG")

data = []

for i in temp1:
    image_org = cv2.imread(i)
    data.append([image_org, [1,0,0,0,0]])

random.shuffle(data)

for i in temp2:
    image_org = cv2.imread(i)
    data.append([image_org, [0,1,0,0,0]])

random.shuffle(data)

for i in temp3:
    image_org = cv2.imread(i)
    data.append([image_org, [0,0,1,0,0]])

random.shuffle(data)

for i in temp4:
    image_org = cv2.imread(i)
    data.append([image_org, [0,0,0,1,0]])

random.shuffle(data)

for i in temp5:
    image_org = cv2.imread(i)
    data.append([image_org, [0,0,0,0,1]])

random.shuffle(data)

X = []
Y = []

test_x = []
test_y = []

random.shuffle(data)

train = data[:-300]
test = data[-300:]

random.shuffle(train)
random.shuffle(test)

for i in train:
    X.append(np.array(i[0]))
    Y.append(np.array(i[1]))

for i in test:
    test_x.append(np.array(i[0]))
    test_y.append(np.array(i[1]))

X = np.array(X).reshape(-1,WIDTH,HEIGHT,1)
Y = np.array(Y)
test_x = np.array(test_x).reshape(-1,WIDTH,HEIGHT,1)
test_y = np.array(test_y)



    #X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
    #Y = [i[1] for i in train] cmd

    #test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
    #test_y = [i[1] for i in test]

model.fit(X, Y, n_epoch= EPOCHS, validation_set=(test_x, test_y),
            snapshot_step=2500, show_metric=True)

model.save(MODEL_NAME)
