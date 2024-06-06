#Combining Datasets
#The following functions load specific datasets and concatenate them:

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
