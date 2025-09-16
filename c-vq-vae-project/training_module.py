import torch


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # if not running in notebook
            return False
    except Exception:
        return False
    return True


if in_notebook(): # different progress bar for notebook and CLI
    from tqdm.notebook import trange, tqdm
else:
    from tqdm import trange, tqdm


def train_model(vae_net, train_loader, device, optimizer, scaler, lr_scheduler, vq_nepoch, patience):
    recon_loss_log = []
    qv_loss_log = []

    min_delta = 0.0
    best_loss = float("inf")
    counter = 0
    end_epoch = vq_nepoch

    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

    pbar = trange(0, vq_nepoch, leave=False, desc="Epoch")
    for epoch in pbar:
        train_loss = 0
        vae_net.train()
        for i, data in enumerate(tqdm(train_loader, leave=False, desc="Training")):
            image = data[0].to(device)
            is_block = data[1].to(device)
            type_ = data[2].to(device)
            colors = data[3].to(device)

            with torch.amp.autocast(autocast_device):
                recon_data, vq_loss, quantized = vae_net(image, is_block=is_block, type_=type_, colors=colors)
                recon_loss = (recon_data - image).pow(2).mean()
                loss = vq_loss + recon_loss

            optimizer.zero_grad()
            if device == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            recon_loss_log.append(recon_loss.item())
            qv_loss_log.append(vq_loss.item())
            train_loss += recon_loss.item()

        lr_scheduler.step()

        vae_net.eval()

        pbar.set_postfix_str(f"Train: {train_loss / len(train_loader):.4f}")

        if recon_loss < best_loss - min_delta:
            best_loss = recon_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                end_epoch = epoch + 1
                break
    return recon_loss_log, qv_loss_log, end_epoch