import shutil
from pathlib import Path

def combine_results():
    base_dir = Path("../data/CIFAR10GS")
    final_dir = base_dir / "combined"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine training set results
    train_dir = final_dir / "train"
    train_dir.mkdir(exist_ok=True)
    
    for machine_id in range(10):  # 10 machines for training set
        machine_dir = base_dir / f"machine_{machine_id}/train"
        for pt_file in machine_dir.glob("*.pt"):
            shutil.copy2(pt_file, train_dir / pt_file.name)
            
    # Combine test set results
    test_dir = final_dir / "test"
    test_dir.mkdir(exist_ok=True)
    
    for machine_id in range(2):  # 2 machines for test set
        machine_dir = base_dir / f"machine_{machine_id}/test"
        for pt_file in machine_dir.glob("*.pt"):
            shutil.copy2(pt_file, test_dir / pt_file.name)

if __name__ == "__main__":
    combine_results() 