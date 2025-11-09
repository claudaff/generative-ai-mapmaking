# Generative AI in Map-Making: A Technical Exploration and Its Implications for Cartographers

[Paper](https://drive.google.com/file/d/16jr8lfLz23eKM1A5qlCg6RzAciimt8Xi/view)

## Abstract

Traditional map-making relies heavily on Geographic Information Systems (GIS), requiring domain expertise and being time-consuming, especially for repetitive tasks. Recent advances in generative AI (GenAI), particularly image diffusion models, offer new opportunities for automating and democratizing the map-making process. However, these models struggle with accurate map creation due to limited control over spatial composition and semantic layout. To address this, we integrate vector data to guide map generation in different styles, specified by the textual prompts. Our model is the first to generate accurate maps in controlled styles, and we have integrated it into a web application to improve its usability and accessibility. We conducted a user study with professional cartographers to assess the fidelity of generated maps, the usability of the web application, and the implications of ever-emerging GenAI in map-making. The findings have suggested the potential of our developed application and, more generally, the GenAI models in helping both non-expert users and professionals in creating maps more efficiently. We have also outlined further technical improvements and emphasized the new role of cartographers to advance the paradigm of AI-assisted map-making.

<img width="2959" height="1030" alt="IdeaOverview" src="https://github.com/user-attachments/assets/49e11b6a-283f-4675-922f-c07cc27f1b02" />

## Instructions

Step-by-step instructions on how to use vector data to control Stable Diffusion with ControlNet for accurate map tile generation.

### 1. Training prerequisites

```sh
git clone https://github.com/lllyasviel/ControlNet.git && cd ControlNet
conda env create -f environment.yml
conda activate control
```
Furthermore, read the official [ControlNet tutorial](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md). No action is required in Steps 0, 1, and 2. Then, in Step 3, one has to decide which version of Stable Diffusion should be used. In our work we employed Stable Diffusion (SD) 1.5. As this version belongs to the group of legacy deprecated SD models, you need to download "v1-5-pruned.ckpt" from [here](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) instead. Alternatively, SD 2.1 ["v2-1_512-ema-pruned.ckpt"](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) or even newer versions could be used. But this was not tested by us. 

Afterwards, run the correct script depending on your chosen version of SD. In our case with SD 1.5, it would be:

```sh
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
```

A ControlNet is now attached to the chosen SD model.

### 2. Dataset creation

As already described in the tutorial, a ControlNet dataset is a triple structure consisting of _target_, _source_, and _prompt_:

In our case, the workflow to create such a dataset looked as follows:

<img width="579" height="310" alt="Screenshot 2025-11-08 at 19 29 15" src="https://github.com/user-attachments/assets/0b55f6a5-ce04-465c-9038-73d594e10798" />

Therefore:

- _target_ is a folder containing raster map tiles (in .png format) of size 512 x 512 pixels. 
- _source_ is a folder containing the corresponding vector data in the form of vector map images (in .png format) of size 512 x 512 pixels. This folder thus contains the input
conditioning images with which Stable Diffusion will be controlled.
- _prompt_ is a .json file linking each image from _target_ and _source_ to a text prompt.

The contents of _target_, _source_, and _prompt.json_ should look as follows:

<img width="1137" height="202" alt="Dataset_Overview" src="https://github.com/user-attachments/assets/cb6248c0-203b-48b5-b18a-8fbc6ca158ef" />

Note:

- This workflow is straightforward in cases where there exists perfectly corresponding vector data. That is usually the case for modern map styles. In the case of historical map styles, where the maps have not yet been vectorized, this procedure becomes more challenging. In our work, we decided to use the modern _Swisstopo_ vector data even for the historical _Siegfried_ and _Old National_ maps by adjusting the vector layers in a way (i.e., by removing certain layers completely) to achieve the best possible alignment.
- It is important to mask all map labels in the raster data AND vector data (see the neon blue areas, corresponding to the mask layer). If text is not masked in the training set, the generated map tiles will be subject to fake labels (i.e., illegible text that consists of made-up letters). Furthermore, niche classes that rarely appear should be masked as well.
- In our work, all tiles were of scale 1:5000. While it is possible to use smaller scales such as 1:25000, the outputs will likely be blurry with inadequate rendering of smaller objects.
- It is also possible to train multiple map styles at once.

### 2. Load dataset 

Open `loadDataset.py` and adjust all three paths so that they point to the correct training data folder containing target, source, and prompt.json.
Then run the script. Optionally check if the dataset was loaded correctly using `dataset_test.py` as a sanity check.

### 3. Training

First, replace the `logger.py` located in the `cldm` folder with the `logger.py` from this repository. There, check the comments and adjust the variables marked with `NEED`.

Open `trainCN.py`, adjust the settings (see our paper and the ControlNet tutorial for reasonable values) and run the script to train ControlNet. To train using Low VRAM Mode, edit the config.py file accordingly.

Note:

- The better the alignment between _target_ and _source_, the better the resulting model.
- Keep in mind that the evaluation loop might take some time to execute each epoch. Therefore, ideally the validation set should not consist of 1000s of tiles. In our work we chose 100 tiles.
- Validation is done by computing the MSE between target (ground-truth) and the generated map tile (model output). This metric only makes sense when training is done using perfectly corresponding vector data!

### 4. Evaluation

After training, adjust and run `evaluateCN.py` to qualitatively evaluate the model on a test set. The generated map tiles are saved as a NumPy array and can then, if needed, be stitched together.

Note:

- When training the model with historical raster data without perfectly corresponding vector data, the generated map tiles can be of poor quality. One way to increase the output quality would be to generate multiple versions of the same tile using different seeds (set `seed = -1` and `num_samples = 6` or any other value larger than 1). Then, using a method of your choice, automatically select the best generated version. In our work we did this automatic selection by employing a segmentation model and also computing the standard deviation of pixel values in the background regions. 

## Models

Our four ControlNet models can be downloaded [here](https://huggingface.co/claudaff/Cartographic-ControlNet/tree/main).

1. `Swisstopo.ckpt`: Specialized model for _Swisstopo style_
2. `OldNational.ckpt`: Specialized model for _Old National style_
3. `Siegfried.ckpt`: Specialized model for _Siegfried style_
4. `Combined.ckpt`: Combined model, capable of generating map tiles in all three styles and used in our web app.

| Class label | RGB color code | Swisstopo | Old National | Siegfried |
|-----------|-----------|-----------|-----------|-----------|
| Background | (255, 255, 255) |✓|✓ | ✓|
| Building | (82, 82, 82) |✓|✓ | ✓|
| Coordinate grid | (237, 240, 64) |✓|✓ | ✓|
| Railway (single track) |  (219, 30, 42) |✓|✓ | ✓|
| Railway (multi track) |  (144, 20, 28) |✓|✓ | ✓|
| Railway bridge | (226, 132, 115) |✓| | |
| Highway |  (247, 128, 30) |✓|✓ | |
| Highway gallery| (231, 119, 28) |✓|✓ | ✓|
| Road | (149, 74, 162) |✓|✓ | ✓|
| Through road | (255, 103, 227) |✓| | |
| Connecting road | (128, 135, 37) |✓| | |
| Path |  (0, 0, 0) |✓|✓ | ✓|
| Depth contour | (63, 96, 132) |✓|✓ | |
| River | (41, 163, 215) |✓|✓ | ✓|
| Lake | (55, 126, 184) |✓|✓ | ✓|
| Stream |  (89, 180, 208) |✓|✓ | ✓|
| Tree | (63, 131, 55) |✓| | |
| Contour line | (164, 113, 88) |✓|✓ | |
| Forest | (77, 175, 74) |✓|✓ | ✓|



## Web application

![DemoVideo (online-video-cutter com) (1)](https://github.com/user-attachments/assets/d420a751-43ad-4ee5-b0f5-269427ec521d)

Run `webapp.py` with `Combined.ckpt` as the underlying model.



