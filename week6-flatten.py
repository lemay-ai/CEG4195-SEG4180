# container: tensorflow
# Outside the course container: pip install icecream tensorflow torch
import tensorflow as tf
import torch
import numpy as np
from icecream import ic

# region: flatten
data = np.array([[[1, 2, 3], [4, 5, 6]]])
dimensions = "(batch_size, height, width)"
ic(data.shape, dimensions)

# Keras: Flatten keeps the batch dimension
tf_tensor = tf.constant(data, tf.float32)
flatten_layer = tf.keras.layers.Flatten()
ic(flatten_layer(tf_tensor))  # (1, 6)

# PyTorch: view(-1) flattens everything
torch_tensor = torch.tensor(data).float()
flat_view = torch_tensor.view(-1)  # (6,)
ic(flat_view)
# Alternatively, use flatten()
flat_fn = torch.flatten(torch_tensor)
ic(torch.equal(flat_view, flat_fn))
# Keep the batch dimension, like Keras
ic(torch.flatten(torch_tensor, start_dim=1))
# endregion
