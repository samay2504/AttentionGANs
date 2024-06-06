#Dataset Preparation
#Loading and Preprocessing Images
#The function load_and_preprocess_images loads and preprocesses images from a specified directory:

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
    dataset = dataset.map(read_file).map(decode_image)
    dataset = dataset.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
