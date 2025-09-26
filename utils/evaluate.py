import torch

criterion = torch.nn.MSELoss

def evaluate(model, dataloader, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
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

