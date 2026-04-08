from share import *

from typing import *
import json
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch
from torch.utils.data import DataLoader

from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

config_file_name = 'config_dual.json'

# Configure environment
environ_config: Optional[Dict[str, str]] = None
loader_config: Optional[Dict[str, Any]] = None
trainer_config: Optional[Dict[str, Any]] = None
if os.path.exists(config_file_name):
    with open(config_file_name, 'r') as f:
        config = json.load(f)
        environ_config = config.get('environ', None)
        loader_config = config.get('loader', None)
        trainer_config = config.get('trainer', None)

if environ_config is not None:
    for key, value in environ_config.items():
        os.environ[key] = value
else:
    # Recommended setting in case OOM occurs.
    # https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# WARNING: TORCH_CPP_LOG_LEVEL will cause single-node execution using gloo to hang!!!
# https://docs.pytorch.org/docs/main/distributed.html#debugging-torch-distributed-applications
# os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = '10000000'
# os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

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

# Loader
dataset = MyDataset()
if loader_config is not None:
    # NOTE: Reduce batch_size here and increase accumulate_grad_batches proportionately if OOM occurs.
    dataloader = DataLoader(dataset, **loader_config)
else:
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
if trainer_config is not None:
    if trainer_config.get('devices', None) is None:
        selected_devices = select_all_cuda_devices()
        if len(selected_devices) > 0:
            trainer_config['devices'] = selected_devices
            trainer_config['accelerator'] = 'gpu'
            trainer_config['strategy'] = DDPStrategy(
                process_group_backend='nccl',
                find_unused_parameters=True,
            )
        if trainer_config.get('strategy', None) is not None:
            trainer_config['strategy'] = DDPStrategy(**trainer_config['strategy'])
        if trainer_config.get('max_epochs', None) is None:
            trainer_config['max_epochs'] = 1
        if trainer_config.get('precision', None) is None:
            trainer_config['precision'] = 'bf16-mixed'
        if trainer_config.get('accumulate_grad_batches', None) is None:
            trainer_config['accumulate_grad_batches'] = 2
    trainer_config['callbacks'] = [logger]  # override callbacks
    trainer = pl.Trainer(**trainer_config)
    print("Configured PyTorch-Lightning trainer from file")
else:
    selected_devices = select_one_cuda_device()
    trainer = pl.Trainer(
        callbacks = [logger],
        max_epochs=1,
        precision='bf16-mixed',
        num_nodes=1, devices=selected_devices, accelerator='gpu',
        accumulate_grad_batches=2,
        strategy='ddp_find_unused_parameters_true',
    )

# Train!
trainer.fit(model, dataloader)
