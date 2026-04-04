from share import *

import json
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch
from torch.utils.data import DataLoader

from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Recommended setting in case OOM occurs.
# https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# If the GPUs are on separate PCIe root complexes, using GLOO instead of NCCL
# To check this run
#     $ lspci -tv 2>/dev/null
# Example:
# -+-[0000:00]-+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex
#  |           +-00.2  Advanced Micro Devices, Inc. [AMD] Milan IOMMU
#  |           +-01.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
#  |           ...
#  |           +-03.1-[01]--+-00.0  NVIDIA Corporation AD102 [GeForce RTX 4090]
#  |           |            \-00.1  NVIDIA Corporation AD102 High Definition Audio Controller
# +-[0000:40]-+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex
#  |           +-00.2  Advanced Micro Devices, Inc. [AMD] Milan IOMMU
#  |           +-01.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
#  |           +-01.1-[41]--+-00.0  NVIDIA Corporation AD102 [GeForce RTX 4090]
#  |           |            \-00.1  NVIDIA Corporation AD102 High Definition Audio Controller
# NOTE: Gloo copies gradients through CPU RAM for synchronization,
# which is slower than NCCL with P2P.
process_group_backend='gloo'
# process_group_backend='nccl'


# Support use of tensor cores
# https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('high')

# Configs
resume_path = './models/control_sd15_ini.ckpt'
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
# NOTE: Reduce batch_size here and increase accumulate_grad_batches proportionately if OOM occurs.
batch_size = 2

# First, use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)


# Print available CUDA devices and select the single device with the most memory
def select_one_cuda_device():
    device = -1
    device_memory = 0
    # WARNING: The device enumeration may not match the system enumeration!
    print(f"Available CUDA devices: {torch.cuda.is_available()}")
    for i in range(torch.cuda.device_count()):
        device_properties = torch.cuda.get_device_properties(i)
        if device_memory < device_properties.total_memory:
            device = i
            device_memory = device_properties.total_memory
        print(f"- Visible CUDA Device {i}: {torch.cuda.get_device_name(i)}, memory = {device_properties.total_memory / 1024**3:.2f} GB")
    print(f"Selected CUDA Device {device}: {torch.cuda.get_device_name(device)}")
    return [device]

def select_all_cuda_devices():
    devices = []
    # WARNING: The device enumeration may not match the system enumeration!
    print(f"Available CUDA devices: {torch.cuda.is_available()}")
    for i in range(torch.cuda.device_count()):
        device_properties = torch.cuda.get_device_properties(i)
        devices.append(i)
        print(f"- Including CUDA Device {i}: {torch.cuda.get_device_name(i)}, memory = {device_properties.total_memory / 1024**3:.2f} GB")
    return devices


# WARNING: Setting strategy='ddp' will result in an error:
# RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step.
# If this is intentional, you must enable the detection of unused parameters in DDP,
# either by setting the string value `strategy='ddp_find_unused_parameters_true'`
# or by setting the flag in the strategy with `strategy=DDPStrategy(find_unused_parameters=True)`.
#
# Setting strategy='ddp_find_unused_parameters_true' works!

# Optionally configure PyTorch-Lightning from a file
if os.path.exists('PLTrainer.json'):
    with open('PLTrainer.json', 'r') as f:
        trainer_config = json.load(f)
    if trainer_config.get('devices', None) is None:
        trainer_config['devices'] = select_one_cuda_device()
        trainer_config['accelerator'] = 'gpu'
    trainer_config['callbacks'] = [logger]
    trainer = pl.Trainer(**trainer_config)
    print("Configured PyTorch-Lightning trainer from file")
else:
    selected_devices = select_all_cuda_devices()
    trainer = pl.Trainer(
        callbacks = [logger],
        max_epochs=1,
        precision='bf16-mixed',
        num_nodes=1, devices=selected_devices, accelerator='gpu',
        accumulate_grad_batches=2,
        strategy=DDPStrategy(
            process_group_backend=process_group_backend,
            find_unused_parameters=True,
        ),
    )

# Train!
trainer.fit(model, dataloader)
