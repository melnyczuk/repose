import argparse

from torch.utils.data import DataLoader

from src.adapters.coco import Coco, CocoDataset
from src.app import Repose

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Repose GAN")
    parser.add_argument(
        "data",
        type=str,
        help="Training data",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Dir for saved samples",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=100,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers",
    )

    args = parser.parse_args()

    train_loader: DataLoader = DataLoader(
        CocoDataset(args.data),
        batch_size=args.batch,
        num_workers=args.workers,
    )

    repose = Repose(data_length=Coco.LENGTH)
    repose.train(train_loader, args.epochs, save_path=args.outdir)
