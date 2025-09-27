# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import csv
import matplotlib.pyplot as plt

"""# Teacher Model"""


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
    def __init__(self, num_classes=10, num_layers=2, input_dim=28, in_channels=1, out_channels=64,
                 linear_hidden_size=256,
                 use_bn=False, use_dropout=False):
        super(LightNN, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.features = nn.Sequential()
        dim_1 = input_dim
        for i in range(num_layers):
            conv_block = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0), nn.ReLU()]
            if self.use_bn:
                conv_block.append(nn.BatchNorm2d(out_channels))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.features.add_module(f"conv{i + 1}", nn.Sequential(*conv_block))
            in_channels = out_channels
            out_channels = max(out_channels // 2, 1)
            dim_1 = math.floor(((dim_1 - 3 + 1) // 2))

        classifier = [
            nn.Linear(int(dim_1 * dim_1 * in_channels), linear_hidden_size),
            nn.ReLU()
        ]
        if self.use_dropout:
            classifier.append(nn.Dropout(0.5))
        classifier.append(nn.Linear(linear_hidden_size, num_classes))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


"""# Training and Evaluation"""


def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return train_losses


def test(model, test_loader, device):
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
    return accuracy


"""# Experiment"""


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms_mnist = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms_mnist)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms_mnist)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Train teacher model
    teacher = DeepNN(num_classes=10).to(device)
    train(teacher, train_loader, epochs=5, learning_rate=0.001, device=device)
    teacher_acc = test(teacher, test_loader, device)
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")

    # Define grid search parameters
    conv_layers = [1, 2, 3]
    conv_number_of_filters_sizes = [2, 8, 16, 32]
    linear_hidden_sizes = [4, 8, 16, 32]
    results = []

    with open("results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['conv_layers', 'filters', 'hidden_size', 'use_bn', 'use_dropout', 'accuracy', 'losses'])

        for layers in conv_layers:
            for filters in conv_number_of_filters_sizes:
                for hidden_size in linear_hidden_sizes:
                    for use_bn in [False, True]:
                        for use_dropout in [False, True]:
                            student = LightNN(num_classes=10, num_layers=layers, out_channels=filters,
                                              linear_hidden_size=hidden_size, use_bn=use_bn,
                                              use_dropout=use_dropout).to(device)
                            losses = train(student, train_loader, epochs=5, learning_rate=0.001, device=device)
                            accuracy = test(student, test_loader, device)
                            results.append((layers, filters, hidden_size, use_bn, use_dropout, accuracy, losses))
                            writer.writerow([layers, filters, hidden_size, use_bn, use_dropout, accuracy, losses])

    # Plot results for optimal student
    best_model = max(results, key=lambda x: x[5])
    print("Best Model:", best_model)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(best_model[6])
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.title("Accuracy: {:.2f}%".format(best_model[5]))
    plt.show()


if __name__ == '__main__':
    main()
