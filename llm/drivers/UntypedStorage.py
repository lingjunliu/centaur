import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    size = input_dict["size"]
    dtype = input_dict["dtype"]
    layout = input_dict.get("layout", torch.strided)
    requires_grad = input_dict.get("requires_grad", False)
    pin_memory = input_dict.get("pin_memory", False)

    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    storage_size = torch.Size(size).numel()
    storage = torch.UntypedStorage(storage_size)
    if not cpu:
        storage = storage.cuda()
    
    # Create a tensor from the storage and initialize it with some values
    tensor = torch.randn(storage_size, device=storage.device, dtype=torch.float32)
    tensor = tensor.reshape(size).to(dtype)
    
    tensor.requires_grad = requires_grad
    
    if not cpu:
        tensor = tensor.cpu()
    
    return {"result": tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    size = input_dict["size"]
    dtype = input_dict["dtype"]
    requires_grad = input_dict.get("requires_grad", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
        
    with tf.device(device_string):

        dtype_map = {
            torch.float32: tf.float32,
            torch.float64: tf.float64,
            torch.float16: tf.float16,
            torch.int32: tf.int32,
            torch.int64: tf.int64,
            torch.int16: tf.int16,
            torch.int8: tf.int8,
            torch.uint8: tf.uint8,
            torch.bool: tf.bool
        }
        
        tf_dtype = dtype_map[dtype]

        # Fill the tensor with random values to match pytorch version
        tensor = tf.Variable(tf.random.normal(size, dtype=tf_dtype), trainable=requires_grad)
        result = tensor.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01
    
    input_data = {
        "size": (2, 3),
        "dtype": torch.float32,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()