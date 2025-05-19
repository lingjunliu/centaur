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

    result = torch.aminmax(input_tensor, dim=dim, keepdim=keepdim)

    if not cpu:
        result = (result.min.cpu(), result.max.cpu())

    return {"min": result.min.numpy(), "max": result.max.numpy()}

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

        if dim is None:
            min_val = tf.reduce_min(input_tensor)
            max_val = tf.reduce_max(input_tensor)
        else:
            min_val = tf.reduce_min(input_tensor, axis=dim, keepdims=keepdim)
            max_val = tf.reduce_max(input_tensor, axis=dim, keepdims=keepdim)

        min_val_np = min_val.numpy()
        max_val_np = max_val.numpy()

    return {"min": min_val_np, "max": max_val_np}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, -3, 5], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["min"], tf_result["min"], atol=A_TOL)
    assert np.allclose(torch_result["max"], tf_result["max"], atol=A_TOL)
    
    input_data = {
        "input": np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.float32),
        "dim": 0,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["min"], tf_result["min"], atol=A_TOL)
    assert np.allclose(torch_result["max"], tf_result["max"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()