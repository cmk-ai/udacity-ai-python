import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import seaborn as sns
from collections import OrderedDict
import argparse
import json
from pathlib import Path
from train import Classifier

def load_checkpoint(filepath):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    
    arch = checkpoint['arch']
    
    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "resnet152":
        model = models.resnet152(pretrained=True)
    else:
        model = models.vgg19_bn(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    hid_units = checkpoint['hidden_size']
    in_units = checkpoint['input_size']
    out_units = checkpoint['output_size']
    if arch == "resnet152":
        model.fc = nn.Sequential(nn.Linear(in_units, hid_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hid_units, out_units),
                                 nn.LogSoftmax(dim=1))
    else:
        model.classifier = Classifier(in_units, hid_units, out_units)

    model.load_state_dict(checkpoint['state_dict'])
    
    #optimizer.load_state_dict(checkpoint['optimizer_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    width = img.size[0]
    height = img.size[1]
    a_ratio = width/height
    #resize, no change in aspect ratio
    if width < height:
        height = int(256/(width/height)) * 100
        width = 256
    else:
        width = int(256/(width/height)) * 100
        height = 256
    
    img.thumbnail((width,height))
    #crop
    left = int(img.size[0]/2) - int(224/2)
    right = int(img.size[0]/2) + int(224/2)
    bottom = int(img.size[1]/2) - int(224/2)
    top = int(img.size[1]/2) + int(224/2)
    img = img.crop((left, bottom, right, top))
    # convert all value to float (0 - 1)
    img = np.array(img)/255
    # Normalization
    img = (img - np.array([0.485, 0.456, 0.406]) ) / np.array([0.229, 0.224, 0.225])
    # transpose
    img = img.transpose((2, 0, 1))
    return img

def predict(image_path, model, device, cat_to_name, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.to(device)
    image.unsqueeze_(0)
    logps = model.forward(image)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    idx_class = {val: key for key, val in model.class_to_idx.items()}
    top_p = top_p.cpu()
    top_class = top_class.cpu()
    np_top_p = top_p.detach().numpy().tolist()[0]
    np_top_class = top_class.detach().numpy().tolist()[0]
    np_top_class_name = [cat_to_name[idx_class[idx]] for idx in np_top_class]
    return np_top_p, np_top_class_name

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='image_path')
    parser.add_argument(dest='checkpoint')
    parser.add_argument('--top_k', dest='top_k', type=int, default=3)
    parser.add_argument('--category_names', dest='category_names', default="cat_to_name.json")
    parser.add_argument('--gpu', dest='gpu', default='cuda')
    args = parser.parse_args()
    
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu
    
    if ".pth" not in checkpoint:
        checkpoint = checkpoint+".pth"
        
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model = load_checkpoint(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() and gpu=='cuda' else 'cpu')
    
    model = model.to(device)
    np_top_p, np_top_class_name = predict(image_path, model,device, cat_to_name, top_k)
    
    
    #print(np_top_class_name[0], np_top_p[0])
    #print(np_top_class[:top_k],"\n")
    print(f"Flower Name: {np_top_class_name[:top_k]} \n"
                    f"Class Probability: {np_top_p[:top_k]}")
    
    
if __name__ == "__main__":
    main()