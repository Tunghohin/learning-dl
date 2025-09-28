import torch
from utils.train import train
from nets.LogisticRegression.data import load_data, prepare_data
from nets.LogisticRegression.LogisticRegression import LogisticRegression
from utils.evaluate import evaluate, compute_accuracy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    train_set, test_set = load_data()
    train_loader = prepare_data(train_set)
    test_loader = prepare_data(test_set)
    input_size = test_set.data.shape[1] - 1

    model = LogisticRegression(input_size=input_size, num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss
    optimizer = torch.optim.Adam
    print(test_set.data.head())

    test_loss = evaluate(model, test_loader, criterion, device=device)
    print(f'Test Loss: {test_loss:.4f}')
    test_accuracy = compute_accuracy(model, test_loader, device=device)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    train(model, train_loader, criterion, optimizer, epochs=1000, lr=0.001, device=device)

    test_loss = evaluate(model, test_loader, criterion, device=device)
    print(f'Test Loss: {test_loss:.4f}')
    test_accuracy = compute_accuracy(model, test_loader, device=device)
    print(f'Test Accuracy: {test_accuracy:.4f}')

