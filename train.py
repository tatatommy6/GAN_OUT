from pathlib import Path
from dataset import CustomDataset
from models.generator import Generator
from models.discriminator import Discriminator
from torch.utils.data import DataLoader, Subset
from utils import set_requires_grad, missing_region_L1, save_preview, save_checkpoint

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
SAMPLE_DIR = ROOT / "samples"

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 5

LR_GEN = 2e-4
LR_DISC = 2e-4

NUM_WORKERS = 4
LAMBDA_RECON = 50.0
SAVE_INTERVAL = 1

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    full_dataset = CustomDataset(DATA_DIR, img_size = IMG_SIZE)
    dataset = Subset(full_dataset, range(min(5000, len(full_dataset))))

    dataloader = DataLoader(dataset, 
                            batch_size = BATCH_SIZE, 
                            shuffle = True,
                            num_workers = NUM_WORKERS,
                            pin_memory = device.type == "cuda",
                            persistent_workers = NUM_WORKERS > 0,
                            drop_last = True)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr = LR_GEN, betas = (0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = LR_DISC, betas = (0.0, 0.9))
    start_epoch = 1

    for epoch in range(start_epoch, EPOCHS + 1):
        generator.train()
        discriminator.train()

        for batch_index, batch in enumerate(dataloader, start = 1):
            real = batch["real"].to(device)
            mask = batch["mask"].to(device)
            masked = batch["masked"].to(device)
            generator_input = batch["generator_input"].to(device)
            #===========training dicriminator===========

            set_requires_grad(discriminator, True)
            optimizer_D.zero_grad(set_to_none = True)

            with torch.no_grad():
                generated = generator(generator_input)

            fake_composite = (
                real * mask + generated * (1.0 - mask)
            )
            real_score = discriminator(real, mask)
            fake_score = discriminator(fake_composite.detach(), mask)

            #hinge GAN loss
            loss_d_real = F.relu(1.0 - real_score).mean()
            loss_d_fake = F.relu(1.0 + fake_score).mean()
            loss_d = loss_d_real + loss_d_fake

            loss_d.backward()
            optimizer_D.step()

            #===========training generator===========

            set_requires_grad(discriminator, False)
            optimizer_G.zero_grad(set_to_none=True)

            generated = generator(generator_input)

            fake_composite = (real * mask + generated * (1.0 - mask))

            fake_score = discriminator(fake_composite, mask)

            loss_g_adv = -fake_score.mean()
            loss_g_recon = missing_region_L1(generated, real, mask)
            loss_g = (loss_g_adv + LAMBDA_RECON * loss_g_recon)
            loss_g.backward()
            optimizer_G.step()

            preview_data = (
                masked.detach(),
                generated.detach(),
                real.detach(),
                mask.detach(),
            )

            if batch_index % 1 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] "
                    f"Batch [{batch_index}/{len(dataloader)}] "
                    f"D: {loss_d.item():.4f} "
                    f"G: {loss_g.item():.4f} "
                    f"Adv: {loss_g_adv.item():.4f} "
                    f"L1: {loss_g_recon.item():.4f}")

        if epoch % SAVE_INTERVAL == 0 and preview_data is not None: # type: ignore
            preview_masked, preview_generated, preview_real, preview_mask = (
                preview_data
            ) # type: ignore

            save_preview(
                preview_masked,
                preview_generated,
                preview_real,
                preview_mask,
                epoch,
            )

            save_checkpoint(
                generator,
                discriminator,
                optimizer_G,
                optimizer_D,
                epoch,
            )

            print(f"{epoch} epoch 저장 완료")

if __name__ == "__main__":
    train()