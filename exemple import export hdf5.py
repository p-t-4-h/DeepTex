import numpy as np
import h5py
a = np.random.random(size=(100,20))
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('dataset_1', data=a)
h5f.close()

h5f = h5py.File('data.h5','r')
b = h5f['dataset_1'][:]
h5f.close()
np.allclose(a,b)
