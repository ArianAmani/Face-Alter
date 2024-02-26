import torch
import torch.nn.functional as F

from model.module import ConvolutionalVariationalAutoencoder
from data.data_loader import get_data_loader
from utils.conf import image_size, patience


def train(dataset_path, device, epochs=10, lr=1e-3, batch_size=32):
    model = ConvolutionalVariationalAutoencoder().to(device)
    train_loader, val_loader = get_data_loader(
        dataset_path, image_size=image_size, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=3, verbose=True)
    best_val_loss = float('inf')
    p = 0
    for epoch in range(epochs):
        train_loss, train_recon_loss, train_kl_loss = train_step(
            model, train_loader, optimizer, device)
        val_loss, val_recon_loss, val_kl_loss = val_step(
            model, val_loader, device)
        
        scheduler.step(val_loss)
        
        print(f'Epoch: {epoch+1}/{epochs}, \n\
            Train Loss: {train_loss:.4f}, Train Recon Loss: {train_recon_loss:.4f}, Train KL Loss: {train_kl_loss:.4f}, \n\
            Val Loss: {val_loss:.4f}, Val Recon Loss: {val_recon_loss:.4f}, Val KL Loss: {val_kl_loss:.4f}')

        # Early Stopping and Checkpoint saver
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, 'checkpoint.pth')
            p = 0
        else:
            p += 1
            if p > patience:
                break


def train_step(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss, recon_loss, kl_divergence = compute_loss(
            data, output, mu, logvar)
        loss.backward()
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_divergence.item()
        total_loss += loss.item()

    total_loss /= len(data_loader)
    total_recon_loss /= len(data_loader)
    total_kl_loss /= len(data_loader)

    return total_loss, total_recon_loss, total_kl_loss


def val_step(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            output, mu, logvar = model(data)
            loss, recon_loss, kl_divergence = compute_loss(
                data, output, mu, logvar)
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_divergence.item()
            total_loss += loss.item()

    total_loss /= len(data_loader)
    total_recon_loss /= len(data_loader)
    total_kl_loss /= len(data_loader)

    return total_loss, total_recon_loss, total_kl_loss


def compute_loss(inputs, out, mu, log_var):
    kl_divergence = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())
    recon_loss = F.mse_loss(inputs, out, size_average=False)
    loss = recon_loss + kl_divergence

    return loss, recon_loss, kl_divergence
