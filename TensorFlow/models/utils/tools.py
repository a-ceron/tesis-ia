import os
import numpy as np

from sklearn.metrics import confusion_matrix

def max_norm(data):
  return data/np.max(data)

def mkdir(path:str):
    """Make a directory if it does not exist.

    :param path: Path to the directory.
    :type path: str
    """
    if not os.path.exists(path):
        os.makedirs(path)

def link(src:str, dst:str):
    """Create a symbolic link.

    :param src: Source path.
    :type src: str
    :param dst: Destination path.
    :type dst: str
    """
    if not os.path.exists(dst):
        os.symlink(src, dst, target_is_directory=True)

def get_images_files(path:str)->list:
    """Get the images files in a directory.

    :param path: Path to the directory.
    :type path: str
    :return: List of images files.
    :rtype: list
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

def get_classes(path:str)->list:
    """Get the classes in a directory.

    :param path: Path to the directory.
    :type path: str
    :return: List of classes.
    :rtype: list
    """
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def get_confusion_matrix(data, n, gen, image_size, shuffle, model, batch_size=32):
    prediction = []
    targets = []

    i = 0
    for x, y in gen.flow_from_directory(data, target_size=image_size, shuffle=shuffle, batch_size=batch_size*2):
        i += 1
        if i % 50 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        prediction.extend(p)
        targets.extend(y)
        if i >= n:
            break

    cm = confusion_matrix(targets, prediction)
    return cm