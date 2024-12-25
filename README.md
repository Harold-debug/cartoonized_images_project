# 🎨 Image Cartoonization with Pix2Pix GAN 🎨

## Project Overview

This project is a school assignment focused on implementing and deploying a **Pix2Pix GAN** for image-to-image translation. The model is used to cartoonize real-world images, transforming them into visually appealing cartoon-style outputs. The project includes:

- Generating a dataset of images with real & cartoon images
- Training the Pix2Pix GAN model.
- Saving and using the best-performing model.
- Providing a user-friendly interface for image cartoonization.

---

## Features

- **Pix2Pix GAN Architecture**: Uses a U-Net-based generator and a PatchGAN discriminator for realistic and effective image-to-image translation.
- **Interactive Web Application**: A Streamlit-based interface allows users to upload images and instantly see cartoonized outputs.
- **Best Model Handling**: The best-performing model checkpoint is saved during training for future use.
- **Training Enhancements**: Includes early stopping and learning rate scheduling for efficient and optimized training.

---

## Installation

### Prerequisites

- Python 3.7+
- Poetry for dependency management

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/cartoonized_images.git
   cd cartoonized_images

2. **Install Dependencies**

Install Poetry if not already installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -

```
3. **Install Project Dependencies**

    ```bash
    poetry install
    ```

4. **Activate the Virtual Environment**

    ```bash
    poetry shell
    ```

## Usage

### Downloading Real Images

You can download a set of real images for training from the following link. Add the folder at your project's root:

[Download Real Images](https://drive.google.com/file/d/1q9Kux4Zhifcx4xTN4A5rm-jPBEhgT6ps/view?usp=sharing)

### Preparing the Dataset

Organize your dataset automatically as follows by running the file `create_cartoon_dataset.py`:

```bash
python create_cartoon_dataset.py
```

```
dataset/
├── train/
│   ├── real/
│   └── cartoon/
└── test/
     ├── real/
     └── cartoon/
```

Each real image should have a corresponding cartoon image. Ensure all images are resized to 256x256 pixels.

### Training the Model

To train the Pix2Pix GAN:

```bash
python train_pix2pix.py
```

This will save the checkpoints and the best-performing model in the `checkpoints_pix2pix/` and `best_model_pix2pix/` directories.

### Download the Model

If you prefer not to train the model yourself, you can download our best-performing pre-trained model from the following link. Add the folder at your project's root:

[Download Model](https://drive.google.com/file/d/1aasSQTYiMpfVCpqhA5EBUbWgb-FHANZm/view?usp=sharing)

### Using the Web Interface

1. **Run the Interface**

    ```bash
    streamlit run interface_app.py
    ```

2. **Upload and Cartoonize Images**
    - Use the file uploader to select a real-world image.
    - Click the ✨ Cartoonize! 🎨 button to generate the cartoon version.
    - View the original and cartoonized images displayed side by side.

## Example Results on Training

![epoch_10](https://github.com/user-attachments/assets/7c727d0a-f24a-483d-bf14-3ec0dc955698)
![epoch_15](https://github.com/user-attachments/assets/d7858c3d-8ac7-4c5b-8244-843b565dc2f1)
![epoch_25](https://github.com/user-attachments/assets/deaaa6f6-a441-4fd8-802f-6e5c0c555c84)



## Dependencies

Key libraries and frameworks used in this project:
- **TensorFlow**: Core deep learning framework.
- **Streamlit**: Interactive web app framework.
- **Poetry**: Dependency management.
- **Pillow & OpenCV**: Image preprocessing tools.

## Interface and Test on new Images

<img width="1470" alt="Screenshot 2024-12-23 at 12 20 22" src="https://github.com/user-attachments/assets/58c27fe7-696f-49da-81fd-a48cb409743a" />

<img width="1199" alt="Screenshot 2024-12-23 at 12 20 59" src="https://github.com/user-attachments/assets/8714bea1-0c26-4157-b1f3-a30627c8c787" />

<img width="1199" alt="Screenshot 2024-12-23 at 12 21 18" src="https://github.com/user-attachments/assets/ed0b81ed-e0b0-46f7-90be-90a6c70c2cde" />

### Video Demo


[![Demo](https://github.com/Harold-debug/cartoonized_images_project/blob/main/demo/Screenshot%202024-12-23%20at%2012.21.18.png)](https://github.com/Harold-debug/cartoonized_images_project/blob/main/demo/demo-clip.mov)

<video src="https://github.com/Harold-debug/cartoonized_images_project/blob/main/demo/demo-clip.mov" >

