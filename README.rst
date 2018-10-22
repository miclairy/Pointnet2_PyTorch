Pointnet2 PyTorch
=================

Partial implemention of `Pointnet2 <https://github.com/charlesq34/pointnet2>`_ written in `PyTorch <http://pytorch.org>`_.

The custom ops used by Pointnet2 are currently **ONLY** supported on the GPU using CUDA.

Building CUDA kernels
---------------------

- ``mkdir build && cd build``
- ``cmake .. && make``

Exampling training
------------------

Two training examples are provided by ``train_sem_seg.py`` and ``train_cls.py``. 

Modified output of pytorch modules and added meta_dataset.py in data to load BPA meta_dataset and part_datasets.py for loading in point data.
