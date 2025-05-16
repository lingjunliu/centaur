import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.utils as utils

    module = input_dict["module"]
    
    if not cpu:
        module = module.cuda()
    
    utils.remove_spectral_norm(module)
    
    if not cpu:
        module = module.cpu()
    
    return {}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    class DummyLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(DummyLayer, self).__init__()
            self.w = self.add_weight(shape=(2, 2), initializer='random_normal', trainable=True)

        def call(self, inputs):
            return tf.matmul(inputs, self.w)

    module = input_dict["module"]

    if not cpu:
        with tf.device('/GPU:0'):
             module.w.assign(np.array([[1,2],[3,4]], dtype=np.float32))
    else:
        module.w.assign(np.array([[1,2],[3,4]], dtype=np.float32))


    return {}

def main():
    A_TOL = 0.01
    # Example input
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    class DummyModule(nn.Module):
        def __init__(self):
            super(DummyModule, self).__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)
    
    torch_module = DummyModule()
    torch.nn.utils.spectral_norm(torch_module.linear)

    class TFDummyLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(TFDummyLayer, self).__init__()
            self.w = self.add_weight(shape=(2, 2), initializer='random_normal', trainable=True)

        def call(self, inputs):
            return tf.matmul(inputs, self.w)
    
    tf_module = TFDummyLayer()
    _ = tf_module(tf.constant([[1.0, 2.0]], dtype=tf.float32))
    
    input_data = {
        "module": torch_module.linear,
    }

    input_data_tf = {
        "module": tf_module,
    }
    # Torch example
    torch_result = torch_version(input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data_tf)

    print("Success")

if __name__ == "__main__":
    main()