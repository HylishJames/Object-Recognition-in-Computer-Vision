import os
import pandas as pd

import matplotlib.pyplot as plt

from mxnet import nd, image
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval

from imageai.Detection import ObjectDetection
import os

saved_params = ''
# Load Model
model_name = "ResNet50_v2"
pretrained = True if saved_params == '' else False
net = get_model(model_name, pretrained=pretrained)
final_list = []
os.chdir("C:\\Users\\CIMB11\\wolf")
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(r"C:\\Users\\CIMB11\\yolo.h5")
detector.loadModel()
custom_objects = detector.CustomObjects(person=True)
if not pretrained:
    net.load_parameters(opt.saved_params)
for direc in os.listdir("C:\\Users\\CIMB11\\wolf"):
    
    folder = {"Customer":direc,"Person": 0,"2 wheeler":0,"4 wheeler":0,"Animal":0,"Food":0,"Gadget":0,"Instrument":0,"Misc":0,"Nature":0,"Sports":0} 
    for filename in os.listdir("C:\\Users\\CIMB11\\wolf\\"+direc+"\\"):
    
        img = image.imread('C:\\Users\\CIMB11\\wolf\\'+direc+'\\'+filename)        
        img = transform_eval(img)
        pred = net(img)

        topK = 8
        ind = nd.topk(pred, k=topK)[0].astype('int')
        #print('The input picture is classified to be')
        objects = []
        for i in range(topK):
            var = net.classes[ind[i].asscalar()]
            if var in li_2wheeler:
                objects.append("2 wheeler")
            if var in li_4wheeler:
                objects.append("4 wheeler")
            if var in li_Animal:
                objects.append("Animal")
            if var in li_Food:
                objects.append("Food")
            if var in li_Gadget:
                objects.append("Gadget")
            if var in li_Instrument:
                objects.append("Instrument")
            if var in li_Misc:
                objects.append("Misc")
            if var in li_Nature:
                objects.append("Nature")
            if var in li_Sports:
                objects.append("Sports")
        
        detections=detector.detectCustomObjectsFromImage(input_image="C:\\Users\\CIMB11\\wolf"+"\\"+direc+'\\'+filename, output_image_path = "C:\\Users\\CIMB11\\pic1.jpg", custom_objects=custom_objects, minimum_percentage_probability=65)
        person_list = []
        for eachObject in detections:
            person_list.append(eachObject["name"])
            
            if person_list:
                objects.append("Person")
        objects = list(set(objects))
       
        for key in objects:
            folder[key] += 1 
    print(folder)
    final_list.append(folder)

os.remove("C:\\Users\\CIMB11\\pic1.jpg")   
df = pd.DataFrame(final_list)
print(df)