import os
import cv2
import numpy as np
def load_images(data_path):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    carPath = data_path + '/car'    #define folderpath
    noneCarPath = data_path + '/non-car'
    
    for object in os.listdir(carPath):
        objPath = os.path.join(carPath, object) #assign the path of every file
        pic = cv2.imread(objPath)   #read as a picture
        
        pic = cv2.resize(pic, (36,16)) #set image as required
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) #grayscale
        
        tempData = (pic,1) #conbine with classification 1
        dataset.append(tempData) #pb to dataset
        
    
    for object in os.listdir(noneCarPath):
        objPath = os.path.join(noneCarPath,object)
        pic = cv2.imread(objPath)
        
        pic = cv2.resize(pic, (36,16)) #set image as required
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) #grayscale
        
        tempData = (pic,0) #conbine with classification 0
        dataset.append(tempData)
    
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset
