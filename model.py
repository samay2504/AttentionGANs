#Model Architecture
#Generator Model

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

#Discriminator Model

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

#Compiling the Models

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

#Training the GAN

batch_size = 32
epochs = 10000
sample_interval = 200

real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in tqdm(range(epochs), desc="Training GAN"):
    real_images = next(iter(all_images.take(1)))
    real_texts = text_data[:batch_size]

    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict([noise, real_texts])

    d_loss_real = discriminator.train_on_batch([real_images, real_texts], real)
    d_loss_fake = discriminator.train_on_batch([fake_images, real_texts], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    sampled_texts = text_data[np.random.randint(0, text_data.shape[0], batch_size)]
    g_loss = gan.train_on_batch([noise, sampled_texts], real)

    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")
