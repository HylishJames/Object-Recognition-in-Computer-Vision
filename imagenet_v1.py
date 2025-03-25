import argparse
import os

import matplotlib.pyplot as plt

from mxnet import nd, image
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval

if False:
	parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
	parser.add_argument('--model', type=str, required=True,
						help='name of the model to use')
	parser.add_argument('--saved-params', type=str, default='',
						help='path to the saved model parameters')
	parser.add_argument('--input-pic', type=str, required=True,
						help='path to the input picture')
	opt = parser.parse_args()
saved_params = ''
# Load Model
model_name = "ResNet50_v2"
pretrained = True if saved_params == '' else False
net = get_model(model_name, pretrained=pretrained)

if not pretrained:
    net.load_parameters(opt.saved_params)
folder = {"2 wheeler":0,"4 wheeler":0,"Animal":0,"Food":0,"Gadget":0,"Instrument":0,"Misc":0,"Nature":0,"Sports":0}  
for filename in os.listdir("C:\\folder\\ujjwal"):
# Load Images
    #filename="131.jpg"
    #print(filename)
    img = image.imread('C:\\\\folder\\\\ujjwal\\\\'+filename)

        # Transform
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

        #print(net.classes[ind[i].asscalar()])
        #print('\t[%s], with probability %.3f.'%
         #   (net.classes[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
    #print(objects)
    objects = list(set(objects))
    #print(objects)
    for key in objects:
        folder[key] += 1 
print(folder)
