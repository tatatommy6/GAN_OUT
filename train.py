from pathlib import Path
from dataset import CustomDataset
from models.generator import Generator
from models.discriminator import Discriminator
from torch.utils.data import DataLoader
from utils import set_requires_grad, missing_region_L1, save_preview, save_checkpoint

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
SAMPLE_DIR = ROOT / "samples"

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 100

LR_GEN = 2e-4
LR_DISC = 2e-4

NUM_WORKERS = 4
LAMBDA_RECON = 50.0
SAVE_INTERVAL = 10

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset = CustomDataset(DATA_DIR, img_size = IMG_SIZE)

    dataloader = DataLoader(dataset, 
                            batch_size = BATCH_SIZE, 
                            shuffle = True,
                            num_workers = NUM_WORKERS,
                            pin_memory = device.type == "mps",
                            persistent_workers = NUM_WORKERS > 0,
                            drop_last = True)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr = LR_GEN, betas = (0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = LR_DISC, betas = (0.0, 0.9))
    start_epoch = 1

    for eopch in range(start_epoch, EPOCHS + 1):
        generator.train()
        discriminator.train()

        for batch_index, batch in enumerate(dataloader, start = 1):
            real = batch["real"].to(device)
            mask = batch["mask"].to(device)
            masked = batch["masked"].to(device)
            generator_input = batch["generator_input"].to(device)
        #===========training dicriminator===========

        set_requires_grad(discriminator, True)