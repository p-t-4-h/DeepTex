In [1]: import numpy as np
In [2]: import h5py
In [3]: a = np.random.random(size=(100,20))
In [4]: h5f = h5py.File('data.h5', 'w')
In [5]: h5f.create_dataset('dataset_1', data=a)
Out[5]: <HDF5 dataset "dataset_1": shape (100, 20), type "<f8">

In [6]: h5f.close()

You can then load that data back in using: '

In [10]: h5f = h5py.File('data.h5','r')
In [11]: b = h5f['dataset_1'][:]
In [12]: h5f.close()

In [13]: np.allclose(a,b)
Out[13]: True
