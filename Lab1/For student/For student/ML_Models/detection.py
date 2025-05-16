import cv2
import matplotlib.pyplot as plt
import numpy as np


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function output the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    coordinate_data = open(data_path)
    #videoPath = data_path.replace('detectData.txt','video.gif')
    
    Coorarray = []
    index = 0
    for line in coordinate_data.readlines():
        Coorarray.append(line.strip().split(' '))
    Coorarray = Coorarray[1:] #coordinates, and remove the first row
    
        
            
    video = cv2.VideoCapture('data/detect/video.gif')
    ret, frame = video.read()
    
    image_result = []  #for image
    allResults = [] #for storing predict results

    
    while ret:
        process_results = []  #labeling,predict_result

        for each_coordinate in Coorarray: #detect car
            x1,y1,x2,y2,x3,y3,x4,y4 = each_coordinate[0],each_coordinate[1],each_coordinate[2],each_coordinate[3],each_coordinate[4],each_coordinate[5],each_coordinate[6],each_coordinate[7]
            #print(x1,' ',y1,' ',x2,' ',y2,' ',x3,' ',y3,' ',x4,' ',y4,'\n')
            processed_image = crop(x1,y1,x2,y2,x3,y3,x4,y4,frame)
            processed_image = cv2.resize(processed_image, (36, 16))
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

            predicted_result = str(clf.classify([processed_image.reshape(-1)]))
            process_results.append(predicted_result + ' ')
        
        
        for i, coordinate in enumerate(Coorarray):
            if process_results[i] == '1 ':
                x1, y1, x2, y2, x3, y3, x4, y4 = int(coordinate[0]), int(coordinate[1]), int(coordinate[2]), int(coordinate[
                    3]), int(coordinate[4]), int(coordinate[5]), int(coordinate[6]), int(coordinate[7])
                points = np.array([[x1, y1], [x2, y2], [x4, y4],
                                [x3, y3], [x1, y1]], np.int32)
                frame = cv2.polylines(frame, [points], False, (0, 255, 0), thickness=3)
                '''
                # img 來源影像
                # pts 座標陣列 ( 使用 numpy 陣列 )
                # isClosed 多邊形是否閉合，True 閉合，False 不閉合
                # color 線條顏色，使用 BGR
                # thickness 線條粗細，預設 1
                '''
        process_results.append('\n') #final write file form
        allResults.append(process_results)
        image_result.append(frame)
        
                
                
        ret, frame = video.read()
        
    cv2.imwrite('test4.png', image_result[0])
    cv2.destroyAllWindows()
        
    with open('ML_Models_pred.txt', 'w') as file:
        for every_result in allResults:
            file.writelines(every_result)            
    
    # End your code (Part 4)
