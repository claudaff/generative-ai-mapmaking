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

- target is a folder containing raster map tiles (in .png format) of size 512 x 512 pixels.
- source is a folder containing the corresponding vector data in the form of vector map images (in .png format) of size 512 x 512 pixels. This folder thus contains the input
conditioning images with which Stable Diffusion will be controlled.
- prompt is a .json file linking each image from target and source to a text prompt.

The contents of target, source and prompt.json should look as follows:

<img width="1137" height="202" alt="Dataset_Overview" src="https://github.com/user-attachments/assets/cb6248c0-203b-48b5-b18a-8fbc6ca158ef" />

Note:

- This workflow is straight-forward in cases where there exists perfectly corresponding vector data. That is usually the case for modern map styles. In the case of historical map styles, where the maps have not yet been vectorized, this procedure becomes more challenging. In our work, we decided to use the modern _Swisstopo_ vector data even for the historical _Siegfried_ and _Old National_ maps by adjusting the vector layers in a way (i.e., by removing certain layers completely) to achieve best possible aligmnent.
- It is important to mask all map labels in the raster data AND vector data (see the neon blue areas, corresponding to the mask layer). If text is not masked in the training set, the generated map tiles will be subject to fake labels (i.e., illegible text that consists of made-up letters). Furthermore, niche classes that rarely appear should be masked as well.
- In our work, all tiles were of scale 1:5000. While it is possible to use smaller scales such as 1:25000, the outputs will likely be blurry with inadequate rendering of smaller objects.
- It is also possible to train multiple map styles at once.

### 2. Load dataset 

Open `loadDataset.py` and adjust all three paths so that they point to the correct training data folder containg target, source and prompt.json.
Then run the script. Optionally check if the dataset was loaded correctly using `dataset_test.py` as a sanity check.

### 3. Training

First, replace the `logger.py` located in the `cldm` folder with the `logger.py` from this repository. There, check the comments and adjust the variables marked with `NEED`.

Open `trainCN.py`, adjust the settings (see our paper and the ControlNet tutorial for reasonable values) and run the script to train ControlNet. To train using Low VRAM Mode, edit the config.py file accordingly.

### 4. Evaluation

After training, adjust and run `evaluateCN.py`.





