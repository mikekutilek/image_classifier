import json, argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
    
def load_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else: #Use vgg13
        model = models.vgg13(pretrained=True)
    classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units']),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(checkpoint['hidden_units'], 102),
                    nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    if pil_image.size[0] > pil_image.size[1]:
        aspect_ratio = pil_image.size[0] / pil_image.size[1]
        pil_image.thumbnail((pil_image,size[0] / aspect_ratio, 256))
    else:
        aspect_ratio = pil_image.size[1] / pil_image.size[0]
        pil_image.thumbnail((256, (pil_image.size[1] / aspect_ratio)))
    
    left = (pil_image.width - 224) / 2
    top = (pil_image.height - 224) / 2
    right = (pil_image.width + 224) / 2
    bottom = (pil_image.height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    np_image = np.array(pil_image) / 255 #Scale color channels
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (np_image - mean)/std
    
    return img.transpose((2, 0, 1))

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, category_file, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    
    image = process_image(image_path)
    tensor_type = torch.cuda.FloatTensor if gpu else torch.FloatTensor
    tensor = torch.from_numpy(image).type(tensor_type).unsqueeze_(0)
    
    ps = torch.exp(model(tensor))
    #print(torch.max(ps, 1))
    top_probs, classes = ps.topk(topk)
    top_classes = []
    for idx in classes.cpu().data.numpy().squeeze():
        top_classes.append(str(idx+1))
    
    if category_file != '':
        with open(category_file, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[i] for i in top_classes]
        
    return top_probs, top_classes

def plot_probabilities(ps, classes):
    # TODO: Display an image along with the top 5 classes
    # Set up plot
    ps = ps.data.numpy().squeeze()
    
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(np.arange(5), ps)
    ax.set_aspect(0.1)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(top_classes, size='small');
    ax.set_title('Class Probability')
    ax.set_xlim(0, 1.1)

    plt.tight_layout()
    
def main():
    parser = argparse.ArgumentParser(description='Neural Network Trainer')
    parser.add_argument('input', help="Choose the path to the image file")
    parser.add_argument('checkpoint', help="Choose the path to model checkpoint")
    parser.add_argument('--top_k', default=5, help="Choose the top K most likely classes")
    parser.add_argument('--category_names', default='', help="Choose a category mapping file")
    parser.add_argument('--gpu', action='store_true', help="Choose whether the gpu will be enabled or not")
    args = parser.parse_args()
    image_path = args.input
    model = load_model(args.checkpoint)
    

    ps, classes = predict(image_path, model, int(args.top_k), args.category_names, args.gpu)
    print("Top {} probabilities and classes: {} {}".format(args.top_k, ps, classes))
    print("Predicted Class: {}".format(classes[0]))
    print("Probability: {}".format(ps.cpu().data.numpy().squeeze()[0]))
    
if __name__ == '__main__':
    main()
    