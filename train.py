# Imports here
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session

def init_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return train_dir, valid_dir, test_dir

def build_model(arch):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else: #Use vgg13
        model = models.vgg13(pretrained=True)
    return model  
    
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

def redefine_classifier(hidden_units):
    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_units, 102),
                    nn.LogSoftmax(dim=1))
    return classifier

def train_network(architecture, lr, hidden_units, epochs, gpu, save_dir, data_dir):
    train_dir, valid_dir, test_dir = init_data(data_dir)
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    #Define model with given architecture
    model = build_model(architecture)
    
    #Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = redefine_classifier(hidden_units)
    device = torch.device("cuda" if gpu else "cpu")
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Step {steps+1}.. "
                      f"Train loss: {running_loss/len(trainloader):.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    model.class_to_idx = train_data.class_to_idx
    torch.save({
                'arch': architecture, 
                'epochs': epochs,
                'hidden_units': hidden_units,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': model.class_to_idx
                }, save_dir)
    
def main():
    parser = argparse.ArgumentParser(description='Neural Network Trainer')
    parser.add_argument('data_directory', help="Choose where to pull the data from")
    parser.add_argument('--save_dir', default='checkpoint.pth', help="Choose where to save the checkpoint to")
    parser.add_argument('--arch', default='vgg16', help="Choose an architecture")
    parser.add_argument('--learning_rate', default=0.003, help="Choose the learning rate of the training algorithm")
    parser.add_argument('--hidden_units', default=4096, help="Choose the number of hidden units the neural network will have")
    parser.add_argument('--epochs', default=5, help="Choose the number of epochs")
    parser.add_argument('--gpu', action='store_true', help="Choose whether the gpu will be enabled or not")
    args = parser.parse_args()
    
    with active_session():
        train_network(args.arch, float(args.learning_rate), int(args.hidden_units), int(args.epochs), args.gpu, args.save_dir, args.data_directory)

if __name__=='__main__':
    main()