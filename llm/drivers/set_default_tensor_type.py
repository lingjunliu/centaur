import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):

    tensor_type = input_dict["tensor_type"]
    
    if not cpu:
        torch.set_default_tensor_type(tensor_type)
        if tensor_type == torch.FloatTensor:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        elif tensor_type == torch.DoubleTensor:
             torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        elif tensor_type == torch.HalfTensor:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
        result = torch.tensor([1.0, 2.0]).cpu()
        torch.set_default_tensor_type(torch.FloatTensor) 
    else:
        torch.set_default_tensor_type(tensor_type)
        result = torch.tensor([1.0, 2.0])
        torch.set_default_tensor_type(torch.FloatTensor)

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):

    tensor_type = input_dict["tensor_type"]
    
    if tensor_type == torch.FloatTensor:
        dtype = tf.float32
    elif tensor_type == torch.DoubleTensor:
        dtype = tf.float64
    elif tensor_type == torch.HalfTensor:
        dtype = tf.float16
    else:
        raise ValueError("Unsupported tensor type")

    if cpu:
        with tf.device("/cpu:0"):
            result = tf.constant([1.0, 2.0], dtype=dtype)
    else:
        with tf.device("/gpu:0"):
            result = tf.constant([1.0, 2.0], dtype=dtype)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "tensor_type": torch.DoubleTensor
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "tensor_type": torch.FloatTensor
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "tensor_type": torch.HalfTensor
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()