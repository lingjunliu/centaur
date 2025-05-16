import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    # input
    x1 = torch.tensor(input["input"])
    x2 = input.get("dim", None)
    keepdim = input.get("keepdim", False)
    dtype = input.get("dtype", None)
    if dtype:
        dtype = torch.tensor(np.array([], dtype=input["dtype"])).dtype
    
    if not cpu:
        x1 = x1.cuda()

    # output
    if dtype is None:
        y = torch.nansum(x1, dim=x2, keepdim=keepdim)
    else:
        y = torch.nansum(x1, dim=x2, keepdim=keepdim, dtype=dtype)

    if not cpu:
        y = y.cpu()

    if y.dim() == 0:
        y = np.float32(y.item())
    else:
        y = y.numpy()        
    
    return {"nansum": y}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # pre-condition
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        
        # input
        x1 = tf.constant(input["input"])
        x2 = input["dim"]
        keepdim = input.get("keepdim", False)
        dtype = input.get("dtype", None)
        
        if dtype is not None:
            dtype = tf.as_dtype(dtype)

        # output        
        y = tf.experimental.numpy.nansum(x1, axis=x2, keepdims=keepdim, dtype=dtype)
        
        return {"nansum": y.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1, float('nan')]),
        "dim": 0,
        "keepdim": False,
        "dtype": np.float32
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results and print 'equal' or 'not equal'
    assert np.array_equal(torch_result["nansum"], tf_result["nansum"]), "Results not equal"
    print("equal")

if __name__ == "__main__":
    main()