from pathlib import Path
from endterm.data_gen.data_gen import generate_flow_dataset
from endterm.training.train_flow import train

PROJECT_ROOT = Path(__file__).resolve().parent
data_path = PROJECT_ROOT / "endterm" / "data"


if __name__ == "__main__":
    print(data_path)

    # File path to check
    target_file = data_path / "flow_train_data.pt"

    if target_file.exists():
        print(f"Skipping dataset generation: '{target_file.name}' already exists in '{data_path}'.")
    else:
        # Ensure the parent directory structure exists before saving
        data_path.mkdir(parents=True, exist_ok=True)
        generate_flow_dataset(save_path=target_file)

    train_flag = True
    
    if train_flag:
        train(target_file)
