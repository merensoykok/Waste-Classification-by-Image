import re
import os
import numpy as np
from PIL import Image

LABELS = {"battery" : 0,
          "cardboard" : 1,
          "clothes" : 2,
          "glass" : 3,
          "metal" : 4,
          "paper" : 5,
          "plastic" : 6,
          "shoes" : 7,
          "trash": 8}

def remove_number_at_end(input_string):
    # Pattern to match one or more digits at the end of the string
    pattern = re.compile(r'\d+$')
    
    # Use regex to substitute the matched pattern with an empty string
    result = re.sub(pattern, '', input_string)
    
    return result


def read_data(data_path:str, flatten_controller:bool = False, image_size:int = 128):
    num_data = len(os.listdir(data_path))
    X = [] #np.zeros((num_data, image_size*image_size*3), dtype=int)
    y = [] #np.zeros(num_data, dtype=int)
    
    for i, filename in enumerate(os.listdir(data_path)):
        # Read the image
        image_path = os.path.join(data_path, filename)
        image = Image.open(image_path)

        # Resize it to 128x128
        if(image.size != (image_size, image_size)):
            image = image.resize(size=(image_size, image_size))

        # Convert to RGB
        image = image.convert("RGB")

        # Convert to numpy array
        image = np.array(image)

        # Flatten the image
        if flatten_controller:
            image = image.flatten()

        # Get the label
        label = filename.split('_')[-1] # Remove aug_i_ part
        label = label.split('.')[0] # Remove .jpg
        label = remove_number_at_end(label) # Remove number at the end
        label = LABELS[label]

        X.append(image)
        y.append(label)

    x_np = np.array(X)
    y_np = np.array(y)


    return x_np, y_np