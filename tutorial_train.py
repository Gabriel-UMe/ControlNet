from share import *

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Support use of tensor cores
# https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('high')

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Print available CUDA devices and select the one with the most memory
# NOTE: This enumeration may not match the system enumeration!
device = -1
device_memory = 0
print(f"Available CUDA devices: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    device_properties = torch.cuda.get_device_properties(i)
    device_memory += device_properties.total_memory
    if device_memory < device_properties.total_memory:
        device = i
        device_memory = device_properties.total_memory
    print(f"- CUDA Visible Device {i}: {torch.cuda.get_device_name(i)}, memory = {device_properties.total_memory / 1024**3:.2f} GB")


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(max_epochs=-1, num_nodes=1, precision=32, callbacks=[logger], devices=[i], accelerator='gpu', strategy='auto')


# Train!
trainer.fit(model, dataloader)
