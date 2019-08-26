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

class Classifier(nn.Module):
    def __init__(self, in_units, hid_units, out_units):
        super().__init__()
        self.fc1 = nn.Linear(in_units, hid_units)
        self.fc2 = nn.Linear(hid_units, out_units)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

def train_model(model, epochs, device, trainloaders, validloaders, optimizer, criterion):
    steps, running_loss, print_every, count = 0, 0, 10, 0
    for epoch in range(epochs):
        for images, labels in trainloaders:
        #print(images.shape, labels.shape)
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    count += 1
                    for images, labels in validloaders:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"{count}.."
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"valid loss: {valid_loss/len(validloaders):.3f}.. "
                    f"valid accuracy: {accuracy/len(validloaders):.3f}")
                running_loss = 0
                model.train()
    return model
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='data_dir')
    parser.add_argument('--save_dir', dest='save_dir', default=".")
    parser.add_argument('--arch', dest='arch', default="vgg19_bn")
    parser.add_argument('--learning_rate', dest='lrn_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', dest='hid_units', type=int, default=4096)
    parser.add_argument('--epochs', dest='epochs', type=int, default=5)
    parser.add_argument('--gpu', dest='gpu', default='cuda')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    lrn_rate = args.lrn_rate
    hid_units = args.hid_units
    epochs = args.epochs
    gpu = args.gpu
    in_units = 0
    out_units = 0
    
    #data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    
    transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])
    train_datasets = datasets.ImageFolder(train_dir, transform=transform)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=transform)
    
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    
    device = torch.device('cuda' if torch.cuda.is_available() and gpu=='cuda' else 'cpu')
    
    
    if arch == "vgg19_bn":
        model = models.vgg19_bn(pretrained=True)
        in_units = int(list(model.classifier.children())[0].in_features)
        
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        in_units = int(model.classifier.in_features)
    elif arch == "resnet152":
        model = models.resnet152(pretrained=True)
        in_units = int(model.fc.in_features)
    else:
        print("The --arch should be vgg19_bn (or) densenet121 (or) resnet152")
    # setting output units according to number of classes in input data
    out_units = len(trainloaders.sampler.data_source.classes)
    
    for param in model.parameters():
        param.requires_grad = False

    criterion = nn.NLLLoss()
    
    if arch == "vgg19_bn":
        model.classifier = Classifier(in_units, hid_units, out_units)
        optimizer = optim.Adam(model.classifier.parameters(), lr=lrn_rate)
    elif arch == "densenet121":
        model.classifier = Classifier(in_units, hid_units, out_units)
        optimizer = optim.Adam(model.classifier.parameters(), lr=lrn_rate)
    elif arch == "resnet152":
        model.fc = nn.Sequential(nn.Linear(in_units, hid_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hid_units, out_units),
                                 nn.LogSoftmax(dim=1))
        optimizer = optim.Adam(model.fc.parameters(), lr=lrn_rate)
    model.to(device)
    model = train_model(model, epochs, device, trainloaders, validloaders, optimizer, criterion)
    
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {
              'arch' : arch,
              'input_size': in_units,
              'output_size': out_units,
              'hidden_size': hid_units,
              #'device' : device,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epochs':epochs,
              'optimizer_dict':optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    
if __name__ == "__main__":
    main()