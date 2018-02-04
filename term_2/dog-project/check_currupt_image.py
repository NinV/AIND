from glob import glob
import random
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
import cv2


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

if __name__ == '__main__':

    # Gather file paths
    dog_images = glob("dogImages/*/*/*")
    human_face_images = glob("lfw/*/*")
    distract_images = glob("AnimTransDistr/*/*")

    image_paths = distract_images+dog_images+human_face_images
    corrupt_images = []
    for path in tqdm(image_paths):
        try:
            tensor = path_to_tensor(path)
        except:
            corrupt_images.append(path)
            continue
    print(corrupt_images)
    with open("list_corrupt_img.txt",'w') as f:
        for path in corrupt_images:
            f.write(path+'\n')



