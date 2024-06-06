This project implements a Generative Adversarial Network (GAN) that generates images from textual descriptions. The GAN consists of a generator and a discriminator model, and is trained on a custom dataset of birds, flowers, and DogsvsCats images.

## Project Structure

/home/samay25/Desktop/Projects/Python(ML)/GANs/Datasets/
├── Birds
│ ├── train
│ │ ├── ostrich
│ │ ├── palila
│ │ ├── ...
│ └── test
│ ├── ostrich
│ ├── palila
│ ├── ...
├── DogsvsCats
│ ├── train
│ └── test
└── Flowers
├── daisy
├── dandelion
├── rose
├── sunflower
└── tulip

Datasets downloaded locally;
1.Flowers Recognition：https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
2.Dogs vs Cats Dataset: https://www.kaggle.com/competitions/dogs-vs-cats
3.Bird: https://www.kaggle.com/datasets/gpiosenka/100-bird-species


## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- tqdm

## Installation

1. Clone this repository:
    ```bash
    git clone <https://github.com/samay2504/GANs-and-Graphs/>
    ```
2. Navigate to the project directory:
    ```bash
    cd Text-to-Image-GAN
    ```
3. Install the required packages:
    ```bash
    pip install tensorflow numpy matplotlib tqdm
    ```

## Usage

1. Ensure the datasets are placed in the correct directory structure as shown above.
2. Run the `main.py` script to train the GAN:
    ```bash
    python main.py
    ```
3. Follow the prompts to enter a text description and generate an image.

## Code Overview

### Loading and Preprocessing Data

The `load_and_preprocess_images` function loads and preprocesses images from the dataset directory. It handles nested subdirectories within the `train` and `test` folders.

### Text Tokenization and Padding

The `custom_tokenizer` and `pad_sequences_custom` functions handle tokenizing and padding text descriptions.

### Model Architecture

- The `build_generator` function constructs the generator model, which takes noise and text embeddings as inputs and generates images.
- The `build_discriminator` function constructs the discriminator model, which takes images and text embeddings as inputs and predicts whether the images are real or fake.

### Training the GAN

The GAN is trained using the `train_GAN` function, which alternates between training the discriminator and the generator.

### Generating Images

The `generate_image` function generates an image from a given text description using the trained generator model.

## Example

To generate an image from a text description:

1. Run the script:
    ```bash
    python text_to_image.py
    ```
2. Enter a text description when prompted, for example:
    ```
    Enter a text description: A bird with yellow wings.
    ```
3. The generated image will be displayed.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.


