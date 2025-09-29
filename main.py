import torch
from utils.train import train
from nets.MLP.data import load_data, prepare_data
from nets.MLP.MLP import MLP
from nets.LeNet5 import LeNet5
from nets.AlexNet import AlexNet
from utils.evaluate import evaluate, compute_accuracy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3, 4, 4, 5])
y = torch.tensor([2, 2, 3, 4, 5, 5])

if __name__ == '__main__':
    train_set, test_set = load_data()
    train_loader = prepare_data(train_set, batch_size=246)
    test_loader = prepare_data(test_set, batch_size=256)

    input_size = 28 * 28
    model = AlexNet(num_classes=10)
    criterion = torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam

    loss = evaluate(model, test_loader, criterion)
    print(f'Initial Loss: {loss:.4f}')
    accuracy = compute_accuracy(model, test_loader)
    print(f'Initial Accuracy: {accuracy:.4f}')

    train(model, train_loader, criterion, optimizer, epochs=10, lr=0.01)

    loss = evaluate(model, test_loader, criterion)
    print(f'Final Loss: {loss:.4f}')
    accuracy = compute_accuracy(model, test_loader)   
    print(f'Accuracy: {accuracy:.4f}')
