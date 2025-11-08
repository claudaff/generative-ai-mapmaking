import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import config
import cv2
import einops
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
from sewar import mse

# Python script extending the original ControlNet logger.py script with a validation loop

# Variables that can be adapted are marked with 'CAN'
# Variables that need to be adapted are marked with 'NEED'



class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None): # 2000 instead of 20
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()


                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

mse_list = []

class Validation(Callback):

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        print("Start Validation Process")
        model = pl_module

        prompt = 'map in swisstopo style' # NEED
        a_prompt = 'best quality, extremely detailed' # Positive prompt. CAN
        n_prompt = '' # Negative prompt. CAN

        image_resolution = 512
        ddim_steps = 20 # CAN
        guess_mode = False # CAN
        strength = 1 # CAN
        scale = 9 # CAN
        seed = 1286028432 # CAN
        eta = 0 # CAN

        num_samples = 1

        path_s = "ValidationSetSwisstopo/source"  # Validation set. NEED
        dir_list_s = os.listdir(path_s)
        source = dir_list_s

        path_t = "ValidationSetSwisstopo/target" # Validation set. NEED
        dir_list_t = os.listdir(path_t)
        target = dir_list_t

        source.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        target.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        lower_bound = np.array([72, 255, 255, 255])  # Label mask color (neon blue in our work). NEED
        upper_bound = np.array([72, 255, 255, 255])  # Label mask color. NEED

        is_train = pl_module.training
        if is_train:

            ddim_sampler = DDIMSampler(model)
            MSE = 0
            count = 0
            model.eval()

            for s, t in zip(source, target):

                input_img = Image.open(path_s + '/' + s)
                input_image = np.array(input_img)
                input_img.close()

                ground_truth = Image.open(path_t + '/' + t)
                gt = np.array(ground_truth)
                ground_truth.close()

                imagemask = cv2.inRange(input_image, lower_bound, upper_bound)

                with torch.no_grad():

                    img = resize_image(HWC3(input_image), image_resolution)
                    H, W, C = img.shape

                    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
                    control = torch.stack([control for _ in range(num_samples)], dim=0)
                    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

                    if seed == -1:

                        seed = random.randint(0, 65535)
                    seed_everything(seed)

                    if config.save_memory:
                        model.low_vram_shift(is_diffusing=False)

                    cond = {"c_concat": [control],
                            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
                    un_cond = {"c_concat": None if guess_mode else [control],
                               "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
                    shape = (4, H // 8, W // 8)

                    if config.save_memory:
                        model.low_vram_shift(is_diffusing=True)

                    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                            [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
                    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                                 shape, cond, verbose=False, eta=eta,
                                                                 unconditional_guidance_scale=scale,
                                                                 unconditional_conditioning=un_cond)

                    if config.save_memory:
                        model.low_vram_shift(is_diffusing=False)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(
                        0, 255).astype(
                        np.uint8)

                    results = [x_samples[i] for i in range(num_samples)]

                output = results[0]

                imagemaskNan = np.where(imagemask == 0, imagemask, np.nan)

                if (np.unique(imagemaskNan).size == 1 and np.unique(imagemaskNan)[0] == 0):  # no mask present

                    count += 1
                    mse_add = mse(output, gt[:, :, :3])
                    MSE += mse_add

                elif (np.unique(imagemaskNan).size == 1 and np.isnan(np.unique(imagemaskNan)[0])):  # only mask

                    MSE = MSE  # no change, and no increase in count

                else:  # masked and non masked regions present -> use np.neanmean

                    count += 1
                    imagemaskNan = cv2.merge((imagemaskNan, imagemaskNan, imagemaskNan))
                    outputt = output + imagemaskNan
                    gtt = gt[:, :, :3] + imagemaskNan

                    mse_add = np.nanmean((gtt.astype(np.float64) - outputt.astype(np.float64)) ** 2)
                    MSE += mse_add
                    # print(np.nanmean((gtt.astype(np.float64) - outputt.astype(np.float64)) ** 2))

            mse_val = MSE / count
            print('MSE: ', mse_val)
            print("Validation Process Over")
            print("===============================================================")
            mse_list.append(round(mse_val))
            print("Summary: ", mse_list)
            metrics = {'val_mse': round(mse_val)}
            self.log_dict(metrics, on_epoch=True)

        if is_train:
            model.train()
            model.to('cuda')
