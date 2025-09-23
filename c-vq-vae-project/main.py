import data_module as dm
import model_module as mm
import training_module as trm
import transformer_module as tfm

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import sys

vae_net = None
tf_generator = None
dataset_loader = None

def prepare_model_for_training(ds, code_book_size, latent_channels, ch, cc, device):
    type_dim = len(ds.iloc[0]['type'])
    colors_dim = len(ds.iloc[0]['colors'])
    model = mm.VQVAE(channel_in=4, latent_channels=latent_channels, ch=ch,
                     code_book_size=code_book_size, commitment_cost=cc,
                     cond_type_dim=type_dim, cond_colors_dim=colors_dim, cond_hidden=256).to(device)
    return model


def run_model_script(device, batch_size, lr, code_book_size, latent_channels, ch, commitment_cost, vq_nepoch, patience, num_workers, train):
    global vae_net, dataset_loader

    # --- Prepare data ---
    dataset_loader = dm.prepare_dataset_loader(ds, batch_size, num_workers)
    # --- Prepare model ---
    vae_net = prepare_model_for_training(ds, code_book_size, latent_channels, ch, commitment_cost, device)
    if train:
        # Setup optimizer
        optimizer = optim.Adam(vae_net.parameters(), lr=lr, weight_decay=1e-5)
        scaler = None
        if device == 'cuda':
            scaler = torch.amp.GradScaler('cuda')
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=vq_nepoch,
                                                            eta_min=0)
        # --- Train model ---
        print("device used: ", device, "\nTraining model...")
        recon_loss_log, qv_loss_log, end_epoch = \
            trm.train_model(vae_net, dataset_loader, device, optimizer, scaler, lr_scheduler,
                            vq_nepoch, patience)

        # --- Evaluate model ---
        vae_net.eval()
        dataiter = iter(dataset_loader)
        test_batch = next(dataiter)
        image = test_batch[0].to(device)
        is_block = test_batch[1].to(device)
        type_ = test_batch[2].to(device)
        colors = test_batch[3].to(device)

        with torch.no_grad():
            recon_data, _, _ = vae_net(image, is_block=is_block, type_=type_, colors=colors)

        # --- Save results ---
        # --- Plot results ---
        plt.figure(figsize=(14, 8))
        plt.axis('off')

        # reconstruction loss
        plt.subplot(2, 2, 1)
        x_train = np.linspace(0, end_epoch, len(recon_loss_log))
        plt.plot(x_train, recon_loss_log)
        plt.title("Reconstruction Loss")
        plt.grid()

        # vq loss
        plt.subplot(2, 2, 3)
        plt.plot(qv_loss_log)
        plt.title("VQ Loss")
        plt.grid()

        # images
        plt.subplot(2, 2, 2)
        plt.title('Original images')
        out = vutils.make_grid(test_batch[0][0:28], normalize=True)
        plt.tight_layout()
        plt.imshow(out.numpy().transpose((1, 2, 0)))

        # reconstructed images
        plt.subplot(2, 2, 4)
        plt.title('Reconstructed images')
        out = vutils.make_grid(recon_data.detach().cpu()[0:28], normalize=True)
        plt.tight_layout()
        plt.imshow(out.numpy().transpose((1, 2, 0)))
        plt.savefig('fig.png')

        print("The model has been saved in .pth files (both the state_dict and full model).")


def run_transformer_script(device, lr, code_book_size, latent_channels, num_epochs, patience, train):
    global tf_generator, dataset_loader
    # Initialize the transformer with condition dimensions
    num_layers = 6
    hidden_size = 256
    num_heads = 8

    tf_generator = tfm.Transformer(
        num_emb=code_book_size + 1,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        latent_channels=latent_channels,
        cond_type_dim=len(ds.iloc[0]['type']),
        cond_colors_dim=len(ds.iloc[0]['colors'])
    ).to(device)

    if train:
        optimizer = optim.Adam(tf_generator.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

        # Mixed precision training
        scaler = None
        if device == 'cuda':
            scaler = torch.amp.GradScaler('cuda')

        loss_fn = nn.CrossEntropyLoss()

        # --- Train transformer ---
        print("device used: ", device, "\nTraining transformer...")
        trm.train_transformer(tf_generator, vae_net, dataset_loader, device, optimizer, scaler, lr_scheduler, num_epochs, patience, loss_fn)


def generate_samples(is_block, type_idx, colors_idx, temperature, test_trained_model, types_len, colors_len):
    try:
        if not test_trained_model:
            vae_net.load_state_dict(torch.load('vae_net_state_dict.pth', map_location=torch.device(device)))
            tf_generator.load_state_dict(torch.load('transformer_state_dict.pth', map_location=torch.device(device)))
        else:
            vae_net.load_state_dict(torch.load('trained_model_state_dict.pth', map_location=torch.device(device)))
            tf_generator.load_state_dict(torch.load('trained_transformer_state_dict.pth', map_location=torch.device(device)))
        return tfm.generate_example_image(tf_generator, vae_net, is_block, type_idx, colors_idx, temperature, types_len, colors_len)
    except RuntimeError as e:
        print(e)
        print("⚠️ Please make sure the parameters in .json file are adjusted to the loaded model as well as the "
              "file_path parameter points to the right dataset (directory) ⚠️")
        return None


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator()
    if device is None:
        device = torch.device('cpu')

    with open('inputs.json', 'r') as file:
        data = json.load(file)

    # --- Download file and process data ---
    file_path = data['file_path']
    ds = dm.download_file(file_path)

    types_len = len(ds['type'].iloc[0])
    colors_len = len(ds['colors'].iloc[0])

    # --- params ---
    # ----------------------------
    is_block = 0
    type_ = torch.zeros(types_len)
    colors = torch.zeros(colors_len)
    temperature = 1.
    for x in [44]:
        type_[x] = 1
    for x in [35]:
        colors[x] = 1
    # ----------------------------

    if ds is None:
        exit(1)
    ds = dm.process_data(ds)
    ds_len = ds.shape[0]

    # --- Get parameters lists ---
    batch_size = data['batch_size']
    lr = data['lr']
    code_book_size = data['code_book_size']
    latent_channels = data['latent_channels']
    ch = data['ch']
    commitment_cost = data['commitment_cost']
    vq_nepoch = data['vq_nepoch']
    tf_nepoch = data['tf_nepoch']
    patience = data['patience']
    num_workers = data['num_workers']

    generate = data['generate']
    test_trained_model = data['test_trained_model']
    train = data['train']

    run_model_script(device, batch_size, lr, code_book_size, latent_channels, ch, commitment_cost, vq_nepoch, patience, num_workers, train)
    run_transformer_script(device, lr, code_book_size, latent_channels, tf_nepoch, patience, train)

    # --- process params ---
    # ----------------------------
    colors_idx = []
    for i, item in enumerate(colors):
        if item == 1:
            colors_idx.append(i)
    type_idx = []
    for i, item in enumerate(type_):
        if item == 1:
            type_idx.append(i)
    # --- output ---
    # ----------------------------
    if generate:
        plt.figure(figsize=(12, 8))
        images = generate_samples(is_block, type_idx, colors_idx, temperature, test_trained_model, types_len, colors_len)
        if images:
            rows = 3
            columns = 7
            for i in range(rows):
                for j in range(columns):
                    plt.subplot(rows, columns, 1 + j + i * columns)
                    plt.imshow(images[j + i * columns])
                    plt.axis('off')
                    plt.tight_layout()
            plt.show()
    # ----------------------------
