from typing import Optional, Union
from pathlib import Path
import torch
from jsonargparse import CLI


def setup(
    num_devices: int = 9,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
    eval_only: bool = False,):
    print("setup")
    print("Input parameters")
    print("devices", num_devices)
    print("train_data_dir", train_data_dir)
    print("val_data_dir", val_data_dir)
    print("precision", precision, type(precision), "tpu", tpu, "resume", resume, "eval_only", eval_only)

    return



if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")


    CLI(setup)
