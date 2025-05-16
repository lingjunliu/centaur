import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    reflection_pad = nn.ReflectionPad2d(padding)
    
    if not cpu:
        reflection_pad = reflection_pad.cuda()

    result = reflection_pad(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        padding = input_dict["padding"]

        shape = tf.shape(input_tensor)
        ndims = len(input_tensor.shape)

        if ndims == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            shape = tf.shape(input_tensor)
            ndims = len(input_tensor.shape)

        if ndims == 4:
            paddings = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
            result = tf.pad(input_tensor, paddings, mode='REFLECT')
        else:
            raise ValueError("Input tensor must be 3D or 4D")

        if len(input_dict["input"].shape) == 3:
            result = tf.squeeze(result, axis=0)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
        "padding": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()