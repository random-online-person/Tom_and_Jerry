import numpy as np
import cv2
from GetKeyPressed import key_check



tom_cascade = cv2.CascadeClassifier("D:/Projects/Emotion_detection/Haar_cascades/tom.xml")
jerry_cascade = cv2.CascadeClassifier("D:/Projects/Emotion_detection/Haar_cascades/jerry.xml")
"""
cap = cv2.VideoCapture('D:/Projects/Emotion_detection/videoplayback.mp4')
Counter = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    size = frame.shape
    Counter += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tom_face = tom_cascade.detectMultiScale(frame, 1.3, 5)
    jerry_face = jerry_cascade.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in tom_face:
        x,y,w,h = expand_rec(x,y,w,h, size)
        temp_img = frame[y:y+h, x:x+h]

        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 2)
        filename = "D:/Projects/Emotion_detection/Youtube_download/Tom/frame_{}_video_{}_tom.JPG".format(Counter, 1)
        cv2.imwrite(filename, cv2.resize(temp_img, (256,256)))



    for (x,y,w,h) in jerry_face:
        x,y,w,h = expand_rec(x,y,w,h, size)
        temp_img = frame[y:y+h, x:x+h]
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 2)
        filename = "D:/Projects/Emotion_detection/Youtube_download/Jerry/frame_{}_video_{}_jerry.JPG".format(Counter,1)
        cv2.imwrite(filename, cv2.resize(temp_img, (256,256)))



    #cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    print("{} frames done, {} much remaining.".format(Counter, 8910 - Counter))

cap.release()
cv2.destroyAllWindows()
"""
def expand_rec(x,y,w,h, size):
    if x > w*0.5:
        x -= 0.5
    else:
        x = 0
    if y > h*0.5:
        y -= h*0.5
    else:
        y = 0
    if x + w*2 <= size[0]:
        w = w * 2
    else:
        w = (size[0] - x)/2
    if y + h*2 <= size[1]:
        h = h * 2
    else:
        h = (size[1] - x)/2

    return int(x),int(y),int(w),int(h)

def get_output_path(keys):
    output_file = "pain"
    if ("Z" in keys):
        output_file = "angry"
    elif "X" in keys:
        output_file = "happy"
    elif "C" in keys:
        output_file = "sad"
    elif "V" in keys:
        output_file = "surprised"
    elif "B" in keys:
        output_file = "unknown"

    return output_file


output_file = "unknown"
Counter = 0
fileName= 'D:/Projects/Emotion_detection/videoplayback (1).mp4'
slomo_frame = 30
cap = cv2.VideoCapture(fileName)       # load the video
while(cap.isOpened()):
    keys = key_check()
    output_file = get_output_path(keys)
    if (output_file != "pain"):
        ret, frame = cap.read()
        if (ret==True):
            size = frame.shape
            Counter += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tom_face = tom_cascade.detectMultiScale(frame, 1.3, 5)
            jerry_face = jerry_cascade.detectMultiScale(frame, 1.3, 5)

            for (x,y,w,h) in tom_face:
                x,y,w,h = expand_rec(x,y,w,h, size)
                temp_img = frame[y:y+h, x:x+h]
                #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 2)
                filename = "D:/Projects/Emotion_detection/data_set/{}/frame_{}_video_{}_tom.JPG".format(output_file,Counter, 2)
                cv2.imwrite(filename, cv2.resize(temp_img, (256,256)))



            for (x,y,w,h) in jerry_face:
                x,y,w,h = expand_rec(x,y,w,h, size)
                temp_img = frame[y:y+h, x:x+h]
                #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 2)
                filename = "D:/Projects/Emotion_detection/data_set/{}/frame_{}_video_{}_jerry.JPG".format(output_file, Counter,2)
                cv2.imwrite(filename, cv2.resize(temp_img, (256,256)))
            print(output_file)
            cv2.imshow('frame',frame)

            if cv2.waitKey(slomo_frame) & 0xFF == ord('q'):
                break

        else:
            break
cap.release()
cv2.destroyAllWindows()
