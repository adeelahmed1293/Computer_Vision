# importing libraries
import cv2
from ultralytics import YOLO
import pandas as pd 
from tracker import Tracker


#Making the Object of Tracker class
tracker=Tracker()

Model=YOLO('yolo11x.pt')

#getting the names of Classes of Coco Dataset
classes=Model.names

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('C:\\Users\\user\\OneDrive\\Desktop\\GPU\\Track_cars\\1.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Read until video is completed
while(cap.isOpened()):
    
# Capture frame-by-frame
    ret, frame = cap.read()
    #taking the result of each Frame
    result=Model.predict(frame)
    #Taking the Bounding Boxes of Result
    if result[0].boxes.data is not None:
        a=result[0].boxes.data
        #Converting the tensors to Numpy array
        a=a.detach().cpu().numpy()
        #Making the data Frame For each Frame in Video
        bx_df=pd.DataFrame(a)
        
        
        #Now Iterating Through the Dataframe and Taking out the Cars only from the Dataset
        list=[]
        for index,data in bx_df.iterrows():
            x1,y1,x2,y2,conf,d=map(int, data)
            c=classes[d]
            if 'car' in c:
                list.append([x1,y1,x2,y2])
        bbox_id = tracker.update(list)
        
        #creating the Bounding Boxes around the cars
        for bbox in bbox_id:
            x3,y3,x4,y4,id=bbox
            cv2.rectangle(frame, (x3,y3),(x4,y4), (0,255,0),2)
            cv2.putText(frame, str(id), (x3,y3), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255))
            
            


                
    # Write the frame with bounding boxes to the output video
    out.write(frame)
    
    if ret == True:
    # Display the resulting frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
         break
     
        
out.release()
cap.release()
cv2.destroyAllWindows()
