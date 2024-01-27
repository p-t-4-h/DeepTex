from PIL import Image, ImageOps
import sys
import numpy as np
from tqdm import tqdm
import threading

#numpy.set_printoptions(threshold=sys.maxsize)


class Image_Processing(threading.Thread):

  def __init__(self, file: str, output=False):
    threading.Thread.__init__(self)
    self.file = file
    self.output = output
    # set a default value
    self.ret = []

  def run(self):
    # Open image
    img = Image.open(f"./Images/{self.file}")
    # Convert in Black and White
    img = ImageOps.invert(img.convert('L'))
    # Convert into NumPy array
    np_d = np.array(img)

    if self.output:
      # Create an output image
      data = Image.fromarray(np_d)
      # Save it
      data.save(f"./Outputs/{self.file}")

      # Reshape the array in 1D
      #np_d_flat = np_d.reshape(-1)
      np_d_flat = np.reshape(np_d, (1, np_d.size))

      # Create an output image
      data = Image.fromarray(np_d_flat)
      # Save it
      data.save(f"./Outputs/flat_{self.file}")

      # Save the NumPy array in .npy file
      #np.save("./Outputs/output.npy", np_d)

    self.ret = np_d


def conv(Input, Filter):
  filter_size = Filter.shape[0]

  #  creating an array to store convolutions (x-m+1, y-n+1)
  convolved = np.zeros(
      ((Input.shape[0] - filter_size) + 1, (Input.shape[1] - filter_size) + 1))

  #  performing convolution
  for i in tqdm(range(Input.shape[0])):
    for j in range(Input.shape[1]):
      try:
        convolved[i, j] = (Input[i:(i + filter_size), j:(j + filter_size)] *
                           Filter).sum()
      except Exception:
        pass

  return convolved


def max_pool(matrice, k_size):
  k_x, k_y = k_size
  m_x, m_y = matrice.shape
  mk = m_x // k_x
  my = m_y // k_y

  return matrice[:mk * k_x, :my * k_y].reshape(mk, k_x, my,
                                               k_y).max(axis=(1, 3))


def train(img_l, output=True):
  for img in img_l:
    t = Image_Processing(img, output)
    t.start()
    np_array = t.ret

    horizontal = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    h = conv(np_array, horizontal)
    v = conv(np_array, vertical)

    data = Image.fromarray(h).convert("L")
    data.save(f"./h_{img}")

    data = Image.fromarray(v).convert("L")
    data.save(f"./v_{img}")

    #Pooling arrays
    k_size = (2, 2)
    h_p = max_pool(h, k_size)
    v_p = max_pool(v, k_size)

    data = Image.fromarray(h_p).convert("L")
    data.save(f"./h_p_{img}")
    data = Image.fromarray(v_p).convert("L")
    data.save(f"./v_p_{img}")
