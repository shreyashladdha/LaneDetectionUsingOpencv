import matplotlib.pylab as plt
import cv2
import numpy as np 

Masked_image=0
Mask=0
Line_image=0
Gray_Image=0
canny_Image=0
Roi_image=0
Image_with_lines=0

def region_of_interest(img, vertices):
        mask=np.zeros_like(img)
        Mask=mask
        match_mask_color= (255,)# * channel_count
        cv2.fillPoly(mask,vertices,match_mask_color)
        masked_image=cv2.bitwise_and(img,mask)
        Masked_image=masked_image
        return masked_image

def draw_lines(img,lines):
    img_new=np.copy(img)
    line_image=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    for line in lines:
        pass
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),thickness=10)
    img= cv2.addWeighted(img,0.8,line_image,1,0.0)
    return img
        
def process(image):
    if image is None:
        return image
    height=image.shape[0]
    width=image.shape[1]

    # region_of_interest_vertices=[
    #     (width/7,height-(height/20)),
    #     (width*0.515, height/1.35),
    #     (width*0.529, height/1.3),
    #     (width*0.70664,height-(height/20))
    # ]
    region_of_interest_vertices=[
        (0,height),(width/2,height/1.70),(width,height)
    ]

    gray_img=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    Gray_Image=gray_img
    Canny_img=cv2.Canny(gray_img,110,130)
    canny_Image=Canny_img   

    roi_image=region_of_interest(Canny_img,np.array([region_of_interest_vertices],np.int32))
    Roi_image= roi_image
    lines=cv2.HoughLinesP(roi_image,rho=2,theta=(np.pi/180),threshold=100,lines=np.array([]),minLineLength=100,maxLineGap=100)
    if lines is not None:
        image_with_lines= draw_lines(image,lines)
        Image_with_lines=image_with_lines
        return image_with_lines
    else:
        return image
    #titles = ['gray_image','Canny_image','Roi image','Image_withline']

    #images = [Gray_Image,canny_Image,Roi_image,image_with_lines]
    #for i in range (4):
    #    plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray')
    #    plt.title(titles[i])
    #    plt.xticks([]),plt.yticks([])
    #plt.show()
    
    #plt.imshow(image_with_lines)
    #plt.show()


cascade_src= 'cars.xml'
cap=cv2.VideoCapture('solidWhiteRight.mp4')
car_cascade=cv2.CascadeClassifier(cascade_src)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame=process(frame)
       # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cars=car_cascade.detectMultiScale(frame,1.09,8)
        for(x,y,w,h) in cars:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(frame,(x+w,y+h),(140,250), font, .5,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

    
cap.release()
cv2.destroyAllWindows()


# cap=cv2.VideoCapture('solidYellowLeft.mp4')
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         frame=process(frame)
#         cv2.imshow('Frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
    
# cap.release()
# cv2.destroyAllWindows()
# image=cv2.imread('Dataset by video recorder/Daytime/beltway/1535.jpg')
# image=process(image)
# plt.imshow(image)
# plt.show()
