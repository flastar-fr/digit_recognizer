import torch
import torch.optim as optim

from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_neurons, output_dim):
        super().__init__()

        self.relu = nn.ReLU()

        layers = [nn.Linear(input_dim, hidden_neurons)]

        for _ in range(hidden_dim - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))

        self.fcs = nn.ModuleList(layers)

        self.fc_output = nn.Linear(hidden_neurons, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        return self.fc_output(x)


def train_model(net, epochs, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.032, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f'[{epoch}/{epochs}] loss: {avg_loss:.3f}')


def test_model(net, test_loader, classes, device):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
