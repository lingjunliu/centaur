import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    dilation = input_dict.get("dilation", 1)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = input_tensor.unfold(dimension=input_dict["dimension"], size=kernel_size, step=stride)
    
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
        kernel_size = input_dict["kernel_size"]
        dilation = input_dict.get("dilation", 1)
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)
        dimension = input_dict["dimension"]

        input_shape = tf.shape(input_tensor)
        input_dim_size = input_shape[dimension]

        output_size = (input_dim_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        padded_input = tf.pad(input_tensor, paddings=[[0, 0]] * dimension + [[padding, padding]] + [[0, 0]] * (len(input_tensor.shape) - dimension - 1))

        indices = tf.range(0, output_size) * stride
        indices = tf.reshape(indices, (-1, 1))
        indices = indices + tf.range(0, kernel_size) * dilation

        indices = tf.reshape(indices, [-1])

        sliced_values = tf.gather(tf.reshape(padded_input, [-1]), indices)

        new_shape = tf.concat([input_shape[:dimension], [output_size, kernel_size], input_shape[dimension+1:]], axis=0)

        result = tf.reshape(sliced_values, new_shape)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "kernel_size": 3,
        "dimension": 0,
        "stride": 1,
        "padding": 0,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()