import numpy as np

def torch_norm_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    p = input.get("p", 'fro')
    dim = input.get("dim", None)
    keepdim = input.get("keepdim", False)
    dtype = input.get("dtype", None)

    if dtype is not None:
        input_tensor = input_tensor.to(dtype)

    # Apply torch.norm
    result = torch.norm(input_tensor, p=p, dim=dim, keepdim=keepdim)
    
    if not cpu:
        result = result.cpu()

    return {"norm": result.numpy()}

def tensorflow_norm_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        p = input.get("p", 'fro')
        dim = input.get("dim", None)
        keepdim = input.get("keepdim", False)

        # Apply TensorFlow equivalent of norm
        if p == 'fro':
            result = tf.norm(input_tensor, ord='euclidean', axis=dim, keepdims=keepdim)
        else:
            result = tf.norm(input_tensor, ord=p, axis=dim, keepdims=keepdim)

        return {"norm": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [-1.0, 1.0, 4.0]], dtype=np.float32),
        "p": 'fro',
        "dim": None,
        "keepdim": False,
        "dtype": None,
    }

    # Torch example
    torch_result = torch_norm_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_norm_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.allclose(torch_result["norm"], tf_result["norm"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()