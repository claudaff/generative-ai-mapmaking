# Generative AI in Map-Making: A Technical Exploration and Its Implications for Cartographers

## Instructions

Step by step instructions on how to use vector data to control Stable Diffusion.

### 1. Training prerequisites

```sh
git clone https://github.com/lllyasviel/ControlNet.git && cd ControlNet
conda env create -f environment.yml
conda activate control
```
Furthermore, read the official [ControlNet tutorial](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md). No action is required in Steps 0, 1, and 2. Then, in Step 3, one has to decide which version of Stable Diffusion should be used. In our work we employed Stable Diffusion (SD) 1.5. As this version belongs to the group of legacy deprecated SD models, you need to download "v1-5-pruned.ckpt" from [here](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) instead. Alternatively, SD 2.1 ["v2-1_512-ema-pruned.ckpt"](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) or even newer versions could be used. But this was not tested by us. 

Afterwards, run the correct script depending on your chosen version of SD. In our case with SD 1.5 it would be:

```sh
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
```

A ControlNet is now attached to the chose SD model.

### 2. Dataset creation

As already described in the tutorial, a ControlNet dataset is a triple structure consisting of the following elements:

- target
- source
- prompt

In our case, the workflow to create such a dataset looked as follows:

<img width="784" height="427" alt="Dataset_Creation" src="https://github.com/user-attachments/assets/16e8da12-5926-40c1-a405-9c3b57ef1b62" />

Therefore:

- target is a folder containing raster map tiles (in .png format) of size 512 x 512 pixels
- source is a folder containing the corresponding vector map tiles (in .png format) of size 512 x 512 pixels
- prompt is a .json file that links each source .png to the corresponding target .png and corresponding prompt






