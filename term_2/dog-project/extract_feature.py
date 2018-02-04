from glob import glob
import random
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor_no_pbar(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)


def extract_feature(paths, feature_shape, batch_size=10):
    from keras.applications.vgg16 import VGG16, preprocess_input
    model = VGG16(weights='imagenet', include_top=False)
    features = np.zeros(feature_shape)
    if len(paths)%batch_size == 0:
        num_batch = len(paths)//batch_size
    else:
        num_batch = len(paths)//batch_size + 1
    with tqdm(total=len(paths), desc="images", unit='images', leave=True) as pbar:
        for i in range(num_batch):
            start = i*batch_size
            stop = (i+1)*batch_size
            tensor = paths_to_tensor_no_pbar(paths[start:stop])
            features[start:stop] = model.predict(preprocess_input(tensor))
            pbar.update(len(paths[start:stop]))
    return features


if __name__ == '__main__':

    # Gather file paths
    dog_images = glob("dogImages/*/*/*")
    human_face_images = glob("lfw/*/*")
    distract_images = glob("AnimTransDistr/*/*")
    print("Number of images in dog dataset", len(dog_images))
    print("Number of images in human dataset", len(human_face_images))
    print("Number of negative samples ", len(distract_images))

    # shuffle file paths
    random.shuffle(dog_images)
    random.shuffle(human_face_images)
    random.shuffle(distract_images)

    # Create file paths in which each class has the same number of images
    num_images_per_class = 3500
    print("Number of images per class using to train", num_images_per_class)
    image_paths = dog_images[0:num_images_per_class] + \
              human_face_images[0:num_images_per_class] + \
              distract_images[0:num_images_per_class]

    # Create label:
    labels = []
    for i in range(num_images_per_class):
        labels.append([1, 0, 0])
    for i in range(num_images_per_class):
        labels.append([0, 1, 0])
    for i in range(num_images_per_class):
        labels.append([0, 0, 1])

    # shuffle again
    data = list(zip(image_paths, labels))
    random.shuffle(data)
    image_paths, labels = zip(*data)
    labels = np.array(labels)

    # extract feature
    features = extract_feature(image_paths, (num_images_per_class*3, 7, 7, 512), 10)
    print(features.shape, features.size)
    np.savez('bottleneck_features/CustomVGG16Data.npz', features=features, labels=labels)

