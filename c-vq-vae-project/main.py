import data_module as dm
import model_module as mm
import training_module as tm

import json
import torch
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def prepare_model_for_training(ds, code_book_size, latent_channels, ch, cc, device):
    type_dim = len(ds.iloc[0]['type'])
    colors_dim = len(ds.iloc[0]['colors'])
    model = mm.VQVAE(channel_in=4, latent_channels=latent_channels, ch=ch,
                     code_book_size=code_book_size, commitment_cost=cc,
                     cond_type_dim=type_dim, cond_colors_dim=colors_dim, cond_hidden=256).to(device)
    return model


def run_script():
    with open('inputs.json', 'r') as file:
        data = json.load(file)

    # --- Download file and process data ---
    file_path = data['file_path']
    ds = dm.download_file(file_path)
    if ds is None:
        return
    ds = dm.process_data(ds)
    ds_len = ds.shape[0]

    # --- Get parameters lists ---
    batch_size_LIST = data['batch_size']
    lr_LIST = data['lr']
    code_book_size_LIST = data['code_book_size']
    latent_channels_LIST = data['latent_channels']
    ch_LIST = data['ch']
    commitment_cost_LIST = data['commitment_cost']
    vq_nepoch_LIST = data['vq_nepoch']
    patience_LIST = data['patience']

    combinations_num = len(batch_size_LIST) * len(lr_LIST) * len(code_book_size_LIST) * len(latent_channels_LIST) \
                        * len(ch_LIST) * len(commitment_cost_LIST) * len(vq_nepoch_LIST) * len(patience_LIST)

    num_workers = data['num_workers']

    combination = 1
    best_loss = 1000000.

    # --- Grid search loop ---
    for batch_size in batch_size_LIST:
        for lr in lr_LIST:
            for code_book_size in code_book_size_LIST:
                for latent_channels in latent_channels_LIST:
                    for ch in ch_LIST:
                        for cc in commitment_cost_LIST:
                            for vq_nepoch in vq_nepoch_LIST:
                                for patience in patience_LIST:
                                    print("combinations attempted: ", combination, "/", combinations_num)
                                    combination+=1

                                    # --- Prepare data ---
                                    device, dataset_loader = dm.prepare_dataset_loader(ds, batch_size, num_workers)
                                    # --- Prepare model ---
                                    vae_net = prepare_model_for_training(ds, code_book_size, latent_channels, ch, cc,
                                                                         device)

                                    # Setup optimizer
                                    optimizer = optim.Adam(vae_net.parameters(), lr=lr, weight_decay=1e-5)
                                    scaler = None
                                    if device == 'cuda':
                                        scaler = torch.amp.GradScaler('cuda')
                                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=vq_nepoch,
                                                                                        eta_min=0)
                                    # --- Train model ---
                                    print("device used: ", device)
                                    recon_loss_log, qv_loss_log, end_epoch = \
                                        tm.train_model(vae_net, dataset_loader, device, optimizer, scaler, lr_scheduler,
                                                       vq_nepoch, patience)
                                    last_epoch_losses = recon_loss_log[-(ds_len // batch_size):]
                                    last_epoch_mean_loss = sum(last_epoch_losses) / len(last_epoch_losses)

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

                                    # --- Save best model's results ---
                                    if best_loss > last_epoch_mean_loss:
                                        best_loss = last_epoch_mean_loss
                                        # --- Plot results ---
                                        plt.figure(figsize=(14, 8))
                                        plt.title("The results for the following combination of parameters:\n" +
                                                  "batch size = " + str(batch_size) + ", lr = " + str(lr) +
                                                  ", codebook size = " + str(code_book_size) + ", latent channels = " +
                                                  str(latent_channels) + ",\nch = " + str(ch) + ", commitment cost = " +
                                                  str(cc) + ", vq_nepoch = " + str(vq_nepoch) + ", patience = " +
                                                  str(patience) + "\n")
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

                                        # --- Save model ---
                                        torch.save(vae_net.state_dict(), 'best_model_state_dict.pth')
                                        torch.save(vae_net, 'best_model_full.pth')

    print("The best model has been saved in .pth files (both the state_dict and full model).\n"
          "Check fig.png file to see the value of the parameters")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_script()
