
import argparse
from pathlib import Path
from torchvision import datasets
from models import GaussianImageTrainer
from constants import CIFAR10_TRANSFORM
from utils import merge_spherical_harmonics, save_gs_data
from configs import Config

def process_dataset_chunk(args):
    # Calculate chunk indices
    if args.train:
        total_size = 50000
        num_chunks = 10
    else:
        total_size = 10000
        num_chunks = 2
        
    chunk_size = total_size // num_chunks
    start_idx = args.chunk_id * chunk_size
    end_idx = start_idx + chunk_size

    # Load dataset
    dataset = datasets.CIFAR10(
        root="../data/CIFAR10/train" if args.train else "../data/CIFAR10/test",
        train=args.train,
        download=True,
        transform=CIFAR10_TRANSFORM,
    )

    # Create output directory
    output_dir = Path(f"../data/CIFAR10GS/machine_{args.chunk_id}/{'train' if args.train else 'test'}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images in chunk
    trainer = GaussianImageTrainer(Config())
    
    for idx in range(start_idx, end_idx):
        image, label = dataset[idx]
        
        # Initialize trainer with current image
        trainer.reinitialize(Config(image=image))
        
        # Train and process splat
        splat = trainer.train()
        splat = merge_spherical_harmonics(splat)
        
        # Save with original dataset index
        save_gs_data(
            image,
            label,
            splat,
            output_dir / f"{idx}.pt"
        )
        
        if (idx - start_idx) % 100 == 0:
            print(f"Processed {idx - start_idx}/{chunk_size} images in chunk {args.chunk_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", type=int, required=True, help="ID of the chunk to process (0-11)")
    parser.add_argument("--train", action="store_true", help="Process training set if set, test set otherwise")
    args = parser.parse_args()
    
    process_dataset_chunk(args) 