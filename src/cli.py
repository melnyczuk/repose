import argparse

from torch.utils.data import DataLoader

from src.repose import Repose
from src.repose.adapters import Coco, CocoDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Repose GAN")
    parser.add_argument(
        "--load",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Load model from weight file",
    )
    parser.add_argument(
        "--train",
        type=str,
        help="Training data",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Dir for saved weights",
    )
    parser.add_argument(
        "--samples",
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

    repose = (
        Repose.load(args.weights)
        if args.load and args.weights
        else Repose(data_length=Coco.LENGTH)
    )

    if args.train and args.weights:
        train_loader: DataLoader = DataLoader(
            CocoDataset(args.train),
            batch_size=args.batch,
            num_workers=args.workers,
        )
        repose.train(train_loader, args.epochs, save_path=args.samples)
        repose.save_weights(args.weights)

    output = Coco.from_tensor(repose.generate())
    print(f"{output=}")
