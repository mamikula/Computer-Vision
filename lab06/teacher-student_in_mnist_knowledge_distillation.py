# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import  torchvision.transforms as transforms
import torchvision.datasets as datasets
import math

"""# Teacher Model"""

# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
        def __init__(self, num_classes=10):
            super(DeepNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(1568, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x            

"""# Student model"""

class LightNN(nn.Module):
        def __init__(self, num_classes=10, num_layers=2, input_dim = 28, in_channels = 1, out_channels = 64, linear_hidden_size = 256):
            super(LightNN, self).__init__()
            self.num_classes = num_classes
            self.num_layers = num_layers
            self.input_dim = input_dim
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.features = nn.Sequential()
            dim_1 = input_dim
            for _ in range(num_layers):
                self.features.add_module(
                    f"conv{_+1}",
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    )
                )
                in_channels = out_channels
                out_channels = int(out_channels/2)
                if out_channels < 1:
                   out_channels = 1

                after_conv = math.floor(((dim_1 + 2*0 -1*(3-1)-1 )/1)+1)
                dim_1 = math.floor(((after_conv + 2*0 -1*(2-1)-1)/2)+1)

                print(dim_1)
            self.classifier = nn.Sequential(
                nn.Linear(int(dim_1*dim_1*in_channels), linear_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(linear_hidden_size, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

"""# Helper functions"""

def main():
    # Check if GPU is available, and if not, use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    transforms_mnist = transforms.Compose([
        transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    #train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms_mnist)
    #test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms_mnist)

    train_dataset = datasets.MNIST(root='C:/Datasets/_base_Datasets_DL/MNIST', train=True, download=True, transform=transforms_mnist)
    test_dataset  = datasets.MNIST(root='C:/Datasets/_base_Datasets_DL/MNIST', train=False, download=True, transform=transforms_mnist)


    #Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)




    def train(model, train_loader, epochs, learning_rate, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                # inputs: A collection of batch_size images
                # labels: A vector of dimensionality batch_size with integers denoting class of each image
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
                # labels: The actual labels of the images. Vector of dimensionality batch_size
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    def test(model, test_loader, device):
        model.to(device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)

        losses = []
        teacher.eval()  # Teacher set to evaluation mode
        student.train() # Student to train mode

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    teacher_logits = teacher(inputs)

                # Forward pass with the student model
                student_logits = student(inputs)

                #Soften the student logits by applying softmax first and log() second
                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
            losses.append((running_loss / len(train_loader)))

        return losses

    """# The experiment

    ## Train the teacher model
    """

    torch.manual_seed(42)
    nn_deep = DeepNN(num_classes=10).to(device)
    train(nn_deep, train_loader, epochs=5, learning_rate=0.001, device=device)
    test_accuracy_deep = test(nn_deep, test_loader, device)

    """## Start the loop"""

    #conv_layers = [1,2,3]
    #conv_number_of_filters_sizes = [1,2,4,8,16,32,64]
    #linear_hidden_sizes = [1,2,4,8,16,32,64]
    
    conv_layers = [2]
    conv_number_of_filters_sizes = [2,8]
    linear_hidden_sizes = [4,8]    

    import csv

    with open("results.csv", 'w', newline='') as file:
       writer = csv.writer(file)
       writer.writerow(['conv_layers', 'cov_filters', 'linear_hidden','teacher_acc','student_acc','teacher_size','student_size','losses'])
       for i in conv_layers:
          for j in conv_number_of_filters_sizes:
             for k in linear_hidden_sizes:
                print("Layers:",i, "| Number of filters:",j, "| Hidden size:",k)
                new_nn_light = LightNN( num_classes=10, num_layers=i, input_dim = 28, in_channels = 1, out_channels = j, linear_hidden_size = k).to(device)

                losses = train_knowledge_distillation(teacher=nn_deep, student=new_nn_light,
                             train_loader=train_loader, epochs=5,
                             learning_rate=0.001, T=2, soft_target_loss_weight=0.75,
                             ce_loss_weight=0.25, device=device)
                test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)
                print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
                print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
                writer.writerow([i, j, k,
                          test_accuracy_deep, test_accuracy_light_ce_and_kd,
                          sum(p.numel() for p in nn_deep.parameters()), sum(p.numel() for p in new_nn_light.parameters()),
                          losses])
                          
if __name__ == '__main__':                          

    main()