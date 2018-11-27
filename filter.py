import face_recognition
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw
from utils import image_resize


print("Engine revving...")
print("Deep Learning activated")
print("Hit 'q' to quit, 'c' to see")

cap = cv2.VideoCapture(0)

logo = cv2.imread('images/meme.png', -1)
watermark = image_resize(logo, height=200)
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)

face_cascade        = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
eyes_cascade        = cv2.CascadeClassifier('xml/frontalEyes35x16.xml')
nose_cascade        = cv2.CascadeClassifier('xml/Nose18x15.xml')
glasses             = cv2.imread("images/glasses.png", -1)
mustache            = cv2.imread('images/mustache.png',-1)

font = cv2.FONT_HERSHEY_SIMPLEX

def apply_invert(frame):
    return cv2.bitwise_not(frame)

rand_num = random.random()

def rand(rand_num,frame):
    if (rand_num % 2) == 0:
        invert = apply_invert(frame)
        return cv2.imshow('EEE 111', invert)
    else:
        return cv2.imshow('EEE 111',frame)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if True:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for (top, right, bottom, left) in (face_locations):
            
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        face_landmarks_list = face_recognition.face_landmarks(frame)

        for face_landmarks in face_landmarks_list:
            pil_image = Image.fromarray(frame)
            d = ImageDraw.Draw(pil_image, 'RGBA')

                # Make the eyebrows into a nightmare
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
        
                # Apply some eyeliner
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

            #for i in range(6):
            #    d.polygon(face_landmarks['bottom_lip'] + face_landmarks['chin'][5:11], fill=(0, 0, 0, 110))
            #d.point(face_landmarks['left_eye'][:4])
            #d.point(face_landmarks['chin'][7:10])
            d.polygon(face_landmarks['bottom_lip'][2:5]+face_landmarks['chin'][7:10],fill=(0,0,110))
            d.polygon(face_landmarks['left_eye'][3:6]+face_landmarks['left_eye'][0:1]+face_landmarks['chin'][5:6],fill=(185,0,0))
            d.polygon(face_landmarks['right_eye'][3:6]+face_landmarks['right_eye'][0:1]+face_landmarks['chin'][11:12],fill=(185,0,0))
      

            frame = np.asarray(pil_image)




    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces           = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # design
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape

    overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
    watermark_h, watermark_w, watermark_c = watermark.shape

    for i in range(0, watermark_h):
        for j in range(0, watermark_w):
            if watermark[i,j][3] != 0:
                offset = 625
                h_offset = frame_h - watermark_h - offset
                w_offset = frame_w - watermark_w - offset
                overlay[h_offset + i, w_offset+ j] = watermark[i,j]
    #face 
    for (x, y, w, h) in faces:
        roi_gray    = gray[y:y+h, x:x+h] # rec
        roi_color   = frame[y:y+h, x:x+h]


        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:

            roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
            glasses2 = image_resize(glasses.copy(), width=ew)

            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    if glasses2[i, j][3] != 0: # alpha 0
                        roi_color[ey + i, ex + j] = glasses2[i, j]


        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (nx, ny, nw, nh) in nose:
            #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
            roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
            mustache2 = image_resize(mustache.copy(), width=nw)

            mw, mh, mc = mustache2.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    #print(glasses[i, j]) #RGBA
                    if mustache2[i, j][3] != 0: # alpha 0
                        roi_color[ny + int(nh/2.0) + i, nx + j] = mustache2[i, j]

    cv2.addWeighted(overlay, 0.25, frame, 1.0, 0, frame)
    frame = cv2.flip(frame, 1)
    cv2.putText(frame,'TRY IT!',(60,455), font, 5,(255,255,255),2,cv2.LINE_AA)
    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow('EEE 111',frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(20) & 0xFF == ord('c'):
        while True:
    	    rand(rand_num,frame)
        if cv2.waitKey(20) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
