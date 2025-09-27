import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def train(
    model, dataloader, 
    criterion, optimizer,
    epochs=100, lr=0.01, device='cuda:0' if torch.cuda.is_available() else 'cpu',
):
    model.to(device)
    opt = optimizer(model.parameters(), lr=lr)
    loss_fn = criterion()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            opt.zero_grad()
            y_pred = model(X_batch)
            batch_loss = loss_fn(y_pred, y_batch)
            batch_loss.backward()
            opt.step()
            writer.add_scalar('Loss/train', batch_loss.item(), epoch)

            epoch_loss += batch_loss.item() * X_batch.size(0)
        
        avg_loss = epoch_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        print(f'Epoch [{epoch+1}/{epochs}] completed.')
