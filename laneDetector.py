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


cap=cv2.VideoCapture('solidWhiteRight.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (960,613))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame=process(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


