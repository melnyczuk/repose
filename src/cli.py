import argparse

from torch.utils.data import DataLoader

from .repose import Repose

parser = argparse.ArgumentParser(description="Run Repose GAN")
parser.add_argument("data", type=str, help="Training data")
parser.add_argument("--batch", help="Batch size")
parser.add_argument("--workers", help="Number of workers")

args = parser.parse_args()

train_loader = DataLoader(
    train_data,
    batch_size=args.batch,
    num_workers=args.workers,
)

repose = Repose()
repose.train(train_loader, 1)
repose.save_samples("train_samples.pkl")
