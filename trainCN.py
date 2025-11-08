from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from loadDataset import MyDataset
from cldm.logger import ImageLogger, Validation
from cldm.model import create_model, load_state_dict

# Python script to train ControlNet
# This script was adopted and adapted from:
# https://github.com/lllyasviel/ControlNet/blob/main/tutorial_train.py

# Configs
resume_path = './models/control_sd15_ini.ckpt'  # adjust to train from pretrained versions e.g., our Swisstopo.ckpt
batch_size = 10  # initially 4
logger_freq = 2000  # defines with which frequency the image log is updated
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="model_{step:02d}_{epoch:02d}_{val_mse}",
        save_top_k=1, monitor='val_mse')
logger = ImageLogger(batch_frequency=logger_freq)
validation_loop = Validation()

trainer = pl.Trainer(callbacks=[logger, checkpoint_callback, validation_loop], precision=16, devices="auto", strategy='ddp_find_unused_parameters_true')  # remove validation_loop and checkpoint_callback if no validation needed
# remove strategy='ddp_find_unused_parameters_true' when training on a single GPU

trainer.fit(model, dataloader)