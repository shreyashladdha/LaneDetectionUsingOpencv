import cv2
cascade_src= 'cars.xml'
video_src= 'solidWhiteRight.mp4'
# video_src= 'solidYellowLeft.mp4'
cap = cv2.VideoCapture(video_src)
car_cascde=cv2.CascadeClassifier(cascade_src)

while True:
    ret,img= cap.read()
    if(type(img)==type(None)):
        break
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cars=car_cascde.detectMultiScale(img,1.1,2)
    for(x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()