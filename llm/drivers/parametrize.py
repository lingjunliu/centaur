import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.utils import parametrize

    module = torch.nn.Linear(5, 10)
    name = input_dict.get("name", "weight")
    parametrization = input_dict["parametrization"]

    if not cpu:
        module = module.cuda()
        parametrization = parametrization.cuda()
    
    parametrize.register_parametrization(module, name, parametrization)
    result = module.weight.detach()
    
    if not cpu:
        result = result.cpu()
    
    parametrize.remove_parametrizations(module, name)
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.linalg import matmul

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        module = layers.Dense(10, input_shape=(5,), use_bias=False)
        _ = module(tf.zeros((1, 5))) 
        name = input_dict.get("name", "weight")
        parametrization = input_dict["parametrization"]

        original_weight = module.kernel
        if callable(parametrization):
            constrained_weight = parametrization(original_weight)
            module.kernel.assign(constrained_weight)
            result = module.kernel.numpy()
        else:
            result = original_weight.numpy()
        result = result.transpose()

    return {"result": result}

def main():
    A_TOL = 0.01
    # Example input
    import torch
    import tensorflow as tf
    from torch.nn.utils import parametrize
    from torch import nn

    class MyReparametrization(nn.Module):
        def forward(self, X):
            return X * 2

        def parametrization(self, X):
            return X / 2

    class TFMyConstraint(tf.keras.constraints.Constraint):
        def __call__(self, w):
            return w * 2
    
    input_data = {
        "name": "weight",
        "parametrization": TFMyConstraint()
    }

    # Torch example
    class TorchMyConstraint(nn.Module):
        def forward(self, w):
            return w * 2
    torch_input_data = {
        "name": "weight",
        "parametrization": TorchMyConstraint()
    }
    torch_result = torch_version(torch_input_data)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()