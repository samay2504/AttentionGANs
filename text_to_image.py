import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

datasets_path = '/home/samay25/Desktop/Projects/Python(ML)/GANs/Datasets'


# Efficiently load and preprocess images using tf.data.Dataset
def load_and_preprocess_images(image_dir, target_size=(64, 64)):
    def decode_image(image_string):
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, target_size)
        image = image / 255.0
        return image

    def read_file(file_path):
        return tf.io.read_file(file_path)

    # Recursively get all image files within the directory and subdirectories
    def get_all_image_files(directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    # Get all image files
    image_files = get_all_image_files(image_dir)

    # Create a dataset from the image file paths
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(read_file).map(decode_image)
    dataset = dataset.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_Birds_images(base_path):
    train_images = load_and_preprocess_images(os.path.join(base_path, 'train'))
    test_images = load_and_preprocess_images(os.path.join(base_path, 'test'))
    return train_images.concatenate(test_images)


def load_Flowers_images(base_path):
    flower_types = ['rose', 'daisy', 'dandelion', 'sunflower', 'tulip']
    datasets = [load_and_preprocess_images(os.path.join(base_path, flower)) for flower in flower_types]

    # Concatenate datasets pairwise
    concatenated_dataset = datasets[0]
    for ds in datasets[1:]:
        concatenated_dataset = concatenated_dataset.concatenate(ds)

    return concatenated_dataset


Birds_images = load_Birds_images(os.path.join(datasets_path, 'Birds'))
Flowers_images = load_Flowers_images(os.path.join(datasets_path, 'Flowers'))
dogsvscats_train_images = load_and_preprocess_images(os.path.join(datasets_path, 'DogsvsCats/train'))
dogsvscats_test_images = load_and_preprocess_images(os.path.join(datasets_path, 'DogsvsCats/test'))

# Combine all images into a single dataset
all_images = Birds_images.concatenate(Flowers_images).concatenate(dogsvscats_train_images).concatenate(
    dogsvscats_test_images)

# Display the first batch of images to ensure proper loading
for images in all_images.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()


# Custom tokenizer and padding
def custom_tokenizer(texts):
    vocab = {}
    sequences = []
    for text in texts:
        sequence = []
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1
            sequence.append(vocab[word])
        sequences.append(sequence)
    return sequences, vocab


def pad_sequences_custom(sequences, maxlen):
    padded = []
    for sequence in sequences:
        if len(sequence) > maxlen:
            padded.append(sequence[:maxlen])
        else:
            padded.append([0] * (maxlen - len(sequence)) + sequence)
    return np.array(padded)


# Example descriptions (these should be actual descriptions from the datasets)
descriptions = ['A bird with yellow wings.', 'A flower with red petals.', 'A cat with white fur.']

sequences, word_index = custom_tokenizer(descriptions)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences_custom(sequences, max_length)

# Repeat the sequences to match the number of images
num_images = len(list(all_images.as_numpy_iterator()))
text_data = np.tile(padded_sequences, (num_images // len(descriptions) + 1, 1))[:num_images]
print(f'Text data shape: {text_data.shape}')


# Generator model
def build_generator(vocab_size, embedding_dim, max_length):
    text_input = tf.keras.Input(shape=(max_length,))
    text_embedding = Embedding(vocab_size, embedding_dim)(text_input)
    text_lstm = LSTM(256)(text_embedding)

    noise_input = tf.keras.Input(shape=(100,))
    x = tf.keras.layers.Concatenate()([noise_input, text_lstm])
    x = Dense(8 * 8 * 256, activation='relu')(x)
    x = Reshape((8, 8, 256))(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh')(x)

    generator = tf.keras.Model([noise_input, text_input], x)
    return generator


# Discriminator model
def build_discriminator(image_shape, vocab_size, embedding_dim, max_length):
    image_input = tf.keras.Input(shape=image_shape)
    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(image_input)
    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = Flatten()(x)

    text_input = tf.keras.Input(shape=(max_length,))
    text_embedding = Embedding(vocab_size, embedding_dim)(text_input)
    text_lstm = LSTM(256)(text_embedding)

    x = tf.keras.layers.Concatenate()([x, text_lstm])
    x = Dense(1, activation='sigmoid')(x)

    discriminator = tf.keras.Model([image_input, text_input], x)
    return discriminator


vocab_size = len(word_index) + 1
embedding_dim = 100
image_shape = (64, 64, 3)

generator = build_generator(vocab_size, embedding_dim, max_length)
discriminator = build_discriminator(image_shape, vocab_size, embedding_dim, max_length)

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False

noise_input = tf.keras.Input(shape=(100,))
text_input = tf.keras.Input(shape=(max_length,))
generated_image = generator([noise_input, text_input])
validity = discriminator([generated_image, text_input])

gan = tf.keras.Model([noise_input, text_input], validity)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train the GAN
batch_size = 32
epochs = 10000
sample_interval = 200

real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in tqdm(range(epochs), desc="Training GAN"):
    # Train Discriminator
    real_images = next(iter(all_images.take(1)))
    real_texts = text_data[:batch_size]

    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict([noise, real_texts])

    d_loss_real = discriminator.train_on_batch([real_images, real_texts], real)
    d_loss_fake = discriminator.train_on_batch([fake_images, real_texts], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    sampled_texts = text_data[np.random.randint(0, text_data.shape[0], batch_size)]
    g_loss = gan.train_on_batch([noise, sampled_texts], real)

    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")


# Function to generate an image from a text description
def generate_image(text_description):
    # Tokenize and pad the text description
    sequence, _ = custom_tokenizer([text_description])
    padded_sequence = pad_sequences_custom(sequence, max_length)

    # Generate noise
    noise = np.random.normal(0, 1, (1, 100))

    # Generate an image using the generator model
    generated_image = generator.predict([noise, padded_sequence])

    # Rescale the generated image from [-1, 1] to [0, 1]
    generated_image = (generated_image + 1) / 2.0

    return generated_image[0]


# Prompt the user for a text description
text_description = input("Enter a text description: ")
generated_image = generate_image(text_description)

# Display the generated image
plt.imshow(generated_image)
plt.axis('off')
plt.show()
