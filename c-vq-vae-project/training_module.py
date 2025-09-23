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

    saved = False

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
            # Save best model
            torch.save(vae_net.state_dict(), 'trained_model_state_dict.pth')
            torch.save(vae_net, 'trained_model_full.pth')
            saved = True
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                end_epoch = epoch + 1
                break
    if not saved:
        # Save final model
        torch.save(vae_net.state_dict(), 'trained_transformer_state_dict.pth')
        torch.save(vae_net, 'trained_transformer_full.pth')
    return recon_loss_log, qv_loss_log, end_epoch


def train_transformer(tf_generator, vae_net, train_loader, device, optimizer, scaler, lr_scheduler, num_epochs, patience, loss_fn):
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    pbar = trange(num_epochs, desc="Epoch")

    # Early stopping
    best_loss = float('inf')
    epochs_no_improve = 0
    saved = False

    for epoch in pbar:
        epoch_loss = 0
        tf_generator.train()

        # DataLoader returns (images, is_block, type_, colors)
        for batch_idx, (images, is_block, type_, colors) in enumerate(tqdm(train_loader, leave=False, desc="Training")):
            # Move data to device
            images = images.to(device)
            is_block = is_block.to(device)
            type_ = type_.to(device)
            colors = colors.to(device)

            # Get discrete tokens from VQ-VAE (no gradients)
            with torch.no_grad():
                _, _, encoding_indices = vae_net.encode(images, is_block, type_, colors)

            # Shift tokens: input is [SOS] + tokens[:-1], target is tokens
            # Use 0 as SOS token (we added +1 to num_emb for this)
            sos_token = torch.zeros_like(encoding_indices[:, 0:1])  # SOS token is 0
            tf_inputs = torch.cat((sos_token, encoding_indices[:, :-1]), 1)
            tf_targets = encoding_indices

            # Forward pass with mixed precision
            with torch.amp.autocast(autocast_device):
                pred = tf_generator(tf_inputs, is_block, type_, colors)
                loss = loss_fn(pred.transpose(1, 2), tf_targets)

            # Backward pass
            optimizer.zero_grad()
            if device == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            # Log progress
            if batch_idx % 100 == 0:
                pbar.set_postfix_str(f'Loss: {epoch_loss / (batch_idx + 1):.4f}')

        # Step scheduler
        lr_scheduler.step()

        # Log epoch loss
        avg_loss = epoch_loss / len(train_loader)
        pbar.set_postfix_str(f'Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(tf_generator.state_dict(), 'trained_transformer_state_dict.pth')
            torch.save(tf_generator, 'trained_transformer_full.pth')
            saved = True
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if not saved:
        # Save final model
        torch.save(tf_generator.state_dict(), 'trained_transformer_state_dict.pth')
        torch.save(tf_generator, 'trained_transformer_full.pth')
