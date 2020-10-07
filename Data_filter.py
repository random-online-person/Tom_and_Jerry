import imagehash
from PIL import Image
import os
import glob


filename = []

temp1 = glob.glob("D:/Projects/Emotion_detection/data_set/angry/*.JPG")
temp2 = glob.glob("D:/Projects/Emotion_detection/data_set/happy/*.JPG")
temp3 = glob.glob("D:/Projects/Emotion_detection/data_set/sad/*.JPG")
temp4 = glob.glob("D:/Projects/Emotion_detection/data_set/surprised/*.JPG")
temp5 = glob.glob("D:/Projects/Emotion_detection/data_set/unknown/*.JPG")

filename = temp1 + temp2 + temp3 + temp4 + temp5

repeat_fileindex = []

for i in range(len(filename) - 1):
    temp1 = imagehash.average_hash(Image.open(filename[i]))
    temp2 = imagehash.average_hash(Image.open(filename[i + 1]))
    cutoff = 10
    if temp1 - temp2 < cutoff:
        repeat_fileindex.append(i)

for i in repeat_fileindex:
    os.remove(filename[i])

print(len(repeat_fileindex))
