import glob

filename = []

temp1 = glob.glob("D:/Projects/Emotion_detection/Crop_imgs/Jerry/*.JPG")
temp2 = glob.glob("D:/Projects/Emotion_detection/Crop_imgs/Tom/*.JPG")
temp3 = glob.glob("D:/Projects/Emotion_detection/Youtube_download/Jerry/*.JPG")
temp4 = glob.glob("D:/Projects/Emotion_detection/Youtube_download/Tom/*.JPG")

filename = temp1 + temp2 + temp3 + temp4

with open('D:/Projects/Emotion_detection/input.txt', 'w') as f:
    for item in filename:
        f.write("%s\n" % item)


#     python app.py -i workdir/input.txt -w workdir -p 8080
