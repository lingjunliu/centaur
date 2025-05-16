import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", None)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    if dim is not None:
        result = torch.min(torch.abs(input_tensor), dim=dim, keepdim=keepdim)
    else:
        result = torch.min(torch.abs(input_tensor))

    if not cpu:
        result = result.cpu()

    if isinstance(result, tuple):
        return {'values': result[0].numpy(), 'indices': result[1].numpy()}
    else:
        return {'values': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        dim = input_dict.get("dim", None)
        keepdim = input_dict.get("keepdim", False)
        
        if dim is not None:
            result = tf.math.reduce_min(tf.math.abs(input_tensor), axis=dim, keepdims=keepdim)
        else:
            result = tf.math.reduce_min(tf.math.abs(input_tensor))
            
        result = result.numpy()

    return {'values': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["values"], tf_result["values"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, -3.0]], dtype=np.float32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["values"], tf_result["values"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, -3.0]], dtype=np.float32),
        "dim": 1,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["values"], tf_result["values"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()