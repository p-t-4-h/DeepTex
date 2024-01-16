from PIL import Image, ImageOps
import sys
import numpy as np
from tqdm import tqdm

#numpy.set_printoptions(threshold=sys.maxsize)

def Image_Processing(file: str, output=False):
    # Open image
    img = Image.open(f"./Images/{file}")
    # Convert in Black and White
    img = ImageOps.invert(img.convert('L'))
    # Convert into NumPy array
    np_d = np.array(img)

    if output:
        # Create an output image
        data = Image.fromarray(np_d)
        # Save it
        data.save(f"./Outputs/{file}")

        # Reshape the array in 1D
        #np_d_flat = np_d.reshape(-1)
        np_d_flat = np.reshape(np_d, (1, np_d.size))

        # Create an output image
        data = Image.fromarray(np_d_flat)
        # Save it
        data.save(f"./Outputs/flat_{file}")

        # Save the NumPy array in .npy file
        #np.save("./Outputs/output.npy", np_d)

    return np_d

def conv(Input, Filter):
    filter_size = Filter.shape[0]

    #  creating an array to store convolutions (x-m+1, y-n+1)
    convolved = np.zeros(((Input.shape[0] - filter_size) + 1, 
                      (Input.shape[1] - filter_size) + 1))
    
    #  performing convolution
    for i in tqdm(range(Input.shape[0])):
      for j in range(Input.shape[1]):
        try:
          convolved[i,j] = (Input[i:(i+filter_size),
                                  j:(j+filter_size)] * Filter).sum()
        except Exception:
          pass
    
    return convolved

np_array = Image_Processing("Numbers.png", output=True)

horizontal = np.array([
                      [1, 1, 1], 
                      [0, 0, 0], 
                      [-1, -1, -1]])
a = conv(np_array, horizontal)
data = Image.fromarray(a).convert("L")
data.save("./img_h.png")

vertical = np.array([
                      [-1, 0, 1], 
                      [-1, 0, 1], 
                      [-1, 0, 1]])
a = conv(np_array, vertical)
data = Image.fromarray(a).convert("L")
data.save("./img_v.png")