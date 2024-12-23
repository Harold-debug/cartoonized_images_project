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

```markdown
3. **Install Project Dependencies**

    ```bash
    poetry install
    ```

4. **Activate the Virtual Environment**

    ```bash
    poetry shell
    ```

## Usage

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

### Using the Web Interface

1. **Run the Interface**

    ```bash
    streamlit run interface_app.py
    ```

2. **Upload and Cartoonize Images**
    - Use the file uploader to select a real-world image.
    - Click the ✨ Cartoonize! 🎨 button to generate the cartoon version.
    - View the original and cartoonized images displayed side by side.

## Example Results

![alt text](samples_pix2pix/epoch_10.png)
![alt text](samples_pix2pix/epoch_15.png)
![alt text](samples_pix2pix/epoch_25.png)
## Dependencies

Key libraries and frameworks used in this project:
- **TensorFlow**: Core deep learning framework.
- **Streamlit**: Interactive web app framework.
- **Poetry**: Dependency management.
- **Pillow & OpenCV**: Image preprocessing tools.
