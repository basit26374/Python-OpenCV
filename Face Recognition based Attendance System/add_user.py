import cv2
import copy

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
haar_face_cascade = cv2.CascadeClassifier('/home/pi/pi-face-recognition/haarcascade_frontalface_default.xml')
img_counter = 0

while True:
    ret, frame = cam.read()
    save_img =  copy.deepcopy(frame)
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = haar_face_cascade.detectMultiScale(temp, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, save_img)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
