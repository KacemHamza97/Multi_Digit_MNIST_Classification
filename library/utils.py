import os
from numpy import array


def rename_image_2_digit(path):
    """rename images using two digits format"""
    for f in os.listdir(path):
        f_new = f"{int(f):02}"
        os.rename(os.path.join(path, f), os.path.join(path, f_new))
        
        
def create_sklearn_data(generator):
    X, y = [], []
    num_samples = len(generator)
    for i, (img, label) in enumerate(generator):
        X.append(img.reshape(-1))
        y.append(int(label[0]))
        if i+1 == num_samples:
            break
    return array(X), array(y)