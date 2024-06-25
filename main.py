import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
import matplotlib.pyplot as plt
from tqdm import tqdm

# Efficiently load and preprocess images using tf.data.Dataset
def load_and_preprocess_images(image_dir, target_size=(64, 64)):
    def decode_image(image_string):
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, target_size)
        image = image / 255.0
        return image

    def read_file(file_path):
        return tf.io.read_file(file_path)

    def get_all_image_files(directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    image_files = get_all_image_files(image_dir)
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(read_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def load_Birds_images(base_path):
    train_images = load_and_preprocess_images(os.path.join(base_path, 'train'))
    test_images = load_and_preprocess_images(os.path.join(base_path, 'test'))
    return train_images.concatenate(test_images)

def load_Flowers_images(base_path):
    flower_types = ['rose', 'daisy', 'dandelion', 'sunflower', 'tulip']
    datasets = [load_and_preprocess_images(os.path.join(base_path, flower)) for flower in flower_types]

    concatenated_dataset = datasets[0]
    for ds in datasets[1:]:
        concatenated_dataset = concatenated_dataset.concatenate(ds)

    return concatenated_dataset

datasets_path = r"D:\PycharmProjects\pythonProject\Datasets"
Birds_images = load_Birds_images(os.path.join(datasets_path, 'Birds'))
Flowers_images = load_Flowers_images(os.path.join(datasets_path, 'Flowers'))
dogsvscats_train_images = load_and_preprocess_images(os.path.join(datasets_path, 'DogsvsCats/train'))
dogsvscats_test_images = load_and_preprocess_images(os.path.join(datasets_path, 'DogsvsCats/test'))

all_images = Birds_images.concatenate(Flowers_images).concatenate(dogsvscats_train_images).concatenate(dogsvscats_test_images)

for images in all_images.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

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

descriptions = ['A bird with yellow wings.', 'A flower with red petals.', 'A cat with white fur.']
sequences, word_index = custom_tokenizer(descriptions)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences_custom(sequences, max_length)

num_images = len(list(all_images.as_numpy_iterator()))
text_data = np.tile(padded_sequences, (num_images // len(descriptions) + 1, 1))[:num_images]
print(f'Text data shape: {text_data.shape}')

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

generator_weights_path = r"D:\PycharmProjects\pythonProject\generator.weights.h5"
discriminator_weights_path = r"D:\PycharmProjects\pythonProject\discriminator.weights.h5"
gan_weights_path = r"D:\PycharmProjects\pythonProject\gan.weights.h5"
epoch_checkpoint_path = r"D:\PycharmProjects\pythonProject\epoch_checkpoint.txt"

def validate_weights(generator, text_description='A bird with yellow wings.'):
    sequence, _ = custom_tokenizer([text_description])
    padded_sequence = pad_sequences_custom(sequence, max_length)

    noise = np.random.normal(0, 1, (1, 100)).astype(np.float32)
    generated_image = generator.predict([noise, padded_sequence])
    generated_image = (generated_image + 1) / 2.0

    plt.imshow(generated_image[0])
    plt.axis('off')
    plt.show()

    # Check if the generated image is reasonable
    if np.mean(generated_image) < 0.1 or np.mean(generated_image) > 0.9:
        return False
    return True

start_epoch = 0
if os.path.exists(generator_weights_path) and os.path.exists(discriminator_weights_path):
    generator.load_weights(generator_weights_path)
    discriminator.load_weights(discriminator_weights_path)
    if validate_weights(generator):
        print("Models loaded and validated, skipping training.")
        if os.path.exists(epoch_checkpoint_path):
            try:
                with open(epoch_checkpoint_path, 'r', encoding='utf-8') as file:
                    start_epoch = int(file.read().strip())
            except (ValueError, UnicodeDecodeError):
                print("Failed to read epoch checkpoint. Starting from epoch 0.")
    else:
        print("Loaded weights are not valid. Proceeding with training.")
else:
    print("No pre-trained weights found. Proceeding with training.")

optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

noise_input = tf.keras.Input(shape=(100,))
text_input = tf.keras.Input(shape=(max_length,))
generated_image = generator([noise_input, text_input])
discriminator.trainable = False
validity = discriminator([generated_image, text_input])
gan = tf.keras.Model([noise_input, text_input], validity)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

batch_size = 32
epochs = 500
sample_interval = 100

class CustomSaveWeightsCallback:
    def __init__(self, generator, discriminator, generator_path, discriminator_path, epoch_path):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_path = generator_path
        self.discriminator_path = discriminator_path
        self.epoch_path = epoch_path

    def on_epoch_end(self, epoch):
        self.generator.save_weights(self.generator_path)
        self.discriminator.save_weights(self.discriminator_path)
        with open(self.epoch_path, 'w', encoding='utf-8') as file:
            file.write(str(epoch))
        print(f"Saved weights and epoch number at epoch {epoch}")

save_callback = CustomSaveWeightsCallback(generator, discriminator, generator_weights_path, discriminator_weights_path, epoch_checkpoint_path)

def generate_image_from_description():
    try:
        text_description = input("Enter a text description: ")
        sequence, _ = custom_tokenizer([text_description])
        padded_sequence = pad_sequences_custom(sequence, max_length)

        noise = np.random.normal(0, 1, (1, 100)).astype(np.float32)
        generated_image = generator.predict([noise, padded_sequence])
        generated_image = (generated_image + 1) / 2.0

        plt.imshow(generated_image[0])
        plt.axis('off')
        plt.show()
    except UnicodeDecodeError as e:
        print(f"Error in input processing: {e}")
        generate_image_from_description()

try:
    for epoch in tqdm(range(start_epoch, epochs)):
        for images in all_images:
            noise = np.random.normal(0, 1, (batch_size, 100))
            idx = np.random.randint(0, text_data.shape[0], batch_size)
            text_batch = text_data[idx]
            generated_images = generator.predict([noise, text_batch])

            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch([images, text_batch], real)
            d_loss_fake = discriminator.train_on_batch([generated_images, text_batch], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = gan.train_on_batch([noise, text_batch], real)

        if epoch % sample_interval == 0:
            noise = np.random.normal(0, 1, (1, 100))
            idx = np.random.randint(0, text_data.shape[0], 1)
            text_sample = text_data[idx]
            generated_image = generator.predict([noise, text_sample])
            generated_image = (generated_image + 1) / 2.0

            plt.imshow(generated_image[0])
            plt.axis('off')
            plt.show()

            save_callback.on_epoch_end(epoch)

except KeyboardInterrupt:
    save_callback.on_epoch_end(start_epoch)
    generate_image_from_description()

def generate_image(text_description):
    sequence, _ = custom_tokenizer([text_description])
    padded_sequence = pad_sequences_custom(sequence, max_length)

    noise = np.random.normal(0, 1, (1, 100)).astype(np.float32)

    generated_image = generator.predict([noise, padded_sequence])

    generated_image = (generated_image + 1) / 2.0

    return generated_image[0]

# Commented out the immediate prompt for user input during initial run
# text_description = input("Enter a text description: ")
# generated_image = generate_image(text_description)

# plt.imshow(generated_image)
# plt.axis('off')
# plt.show()

