import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

seed = 2
model_name = "MRFDPUnet2Plus"
n_channels = 3
n_labels = 1
img_size = 224
batch_size = 16
epochs = 300
learning_rate = 1e-4
early_stopping_patience = 50
num_workers = 2

task_name = "bone"
data_path = Path("datasets") / task_name
train_dataset = data_path / "Train"
val_dataset = data_path / "Val"
test_dataset = data_path / "Test"

session_name = "Session" + str(seed)
save_path = Path(task_name) / model_name / session_name
model_path = save_path / "models"
logger_path = save_path / f"{session_name}.log"
visualize_path = save_path / "test_visualize"
result_path = save_path / "test.result"
best_model_path = model_path / f"best_model-{model_name}.pth.tar"
