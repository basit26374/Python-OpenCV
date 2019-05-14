How to Run the Face Recognition Program


1. After power up the raspberry pi, first of all you may click on command
prompt and type
  source ~/.profile
2. Then you may go into virtual environment using type
workon project
3. Then you go into the directory of your project file, which is named as “piface-recognition” using type on command prompt
cd home/pi/pi-face-recognition
4. After all of that you are ready to run the main python script using type on
command prompt
python pi_face_recognition.py --cascade
haarcascade_frontalface_default.xml \
--encodings encodings.pickle
5. Your attendance program is running now and frame goes appear, if your
want to quit the program, place the mouse cursor on frame and just type
“q”. You will be exit from the program and Libre Calc file of attendance may
generate on your desktop which is named as “Attend.xlsx”


How to Create Database for New Users


1. Just open the “add_user.py” file which is placed on “/home/pi” directory.
2. Run the script using “F5”
3. Frame goes appear on your screen.
4. Then you see that, on some images in video, bounding box were created
across the face of the person, it’s the right images for your dataset.
5. Just press “space bar” for capture the images, for anyone person at least 4
or 5 images must capture.
6. If you take sufficient images in which bounding box were created across the
face, just press “Esc” button. You will be exit from the program
7. Make sure that, there must be one person in one image
8. After that, all images were saved on main directory “/home/pi”.
9. Cut all these images from here and go to the “/home/pi/pi-facerecognition/dataset” folder. Create new folder and keep the name of the
folder according to the person’s name or ID.
10.Open up the command prompt and follow the steps from 1-3 in above
guidance section “Run the Face Recognition Program”
11. The last step for creating bindings of image dataset is to type on command
prompt. After this step you may only follow the steps of “Run the Face
Recognition Program” guidance section.
python encode_faces.py --dataset dataset --
encodings encodings.pickle \
--detection-method hog






This project is modified version of adrian rosebrock Face Recognition tutorial https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
