import cv2

cascPath = "haarcascade_frontalface_default.xml"#frontface pretrained dataset
faceCascade = cv2.CascadeClassifier(cascPath)#using it as the classifier by opencv

video_capture = cv2.VideoCapture(0)#video source(0 for the default camera input)

while True:
    ret, frame = video_capture.read()#capturing every frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#converting it into grayscale(black and white)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )#detecting the faces using the haarcascade dataset
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (186, 85, 211), 8)#drawing a rectangle
    cv2.imshow('Face Detection', frame)#desplaying the vid

video_capture.release()#releasing the capturing
cv2.destroyAllWindows()#closing all of the windows
