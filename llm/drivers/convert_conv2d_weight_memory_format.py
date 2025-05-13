import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input_tensor"])
    memory_format = input_dict["memory_format"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    if memory_format == torch.channels_last:
      input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    elif memory_format == torch.channels_first:
      input_tensor = input_tensor.contiguous(memory_format=torch.channels_first)
    else:
        raise ValueError("Unsupported memory format")
    
    result = input_tensor

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import torch

    input_tensor_np = input_dict["input_tensor"]
    memory_format = input_dict["memory_format"]

    if memory_format == torch.channels_last:
        data_format = "NHWC"
    elif memory_format == torch.channels_first:
        data_format = "NCHW"
    else:
        raise ValueError("Unsupported memory format for TensorFlow conversion")

    if cpu:
        with tf.device("/cpu:0"):
            input_tensor = tf.constant(input_tensor_np)

            if data_format == "NCHW":
                result = tf.transpose(input_tensor, perm=[2, 3, 0, 1])
            else:
                result = tf.transpose(input_tensor, perm=[0, 1, 2, 3])

            result = result.numpy()
    else:
        with tf.device("/gpu:0"):
            input_tensor = tf.constant(input_tensor_np)

            if data_format == "NCHW":
                result = tf.transpose(input_tensor, perm=[2, 3, 0, 1])
            else:
                result = tf.transpose(input_tensor, perm=[0, 1, 2, 3])

            result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input_tensor": np.random.rand(3, 3, 3, 3).astype(np.float32),
        "memory_format": torch.channels_last
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_result_np = torch_result['result']
    tf_result_np = tf_result['result']


    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"
    
    input_data = {
        "input_tensor": np.random.rand(3, 3, 3, 3).astype(np.float32),
        "memory_format": torch.channels_first
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_result_np = torch_result['result']
    tf_result_np = tf_result['result']


    assert np.allclose(torch_result_np, tf_result_np, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()