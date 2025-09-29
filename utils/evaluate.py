import torch

def evaluate(
    model, dataloader, 
    criterion,
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
):
    model.to(device)
    loss_fn = criterion()
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            batch_loss = loss_fn(y_pred, y_batch)
            total_loss += batch_loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def compute_accuracy(model, dataloader, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = torch.argmax(model(X_batch), dim=1).squeeze()
            y_batch = torch.argmax(y_batch, dim=1).squeeze()
            correct += (outputs == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    return accuracy