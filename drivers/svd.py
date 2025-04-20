import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_svd_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    some = input.get("some", True)
    compute_uv = input.get("compute_uv", True)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.svd
    u, s, v = torch.svd(input_tensor, some=some, compute_uv=compute_uv)

    if not cpu:
        u, s, v = u.cpu(), s.cpu(), v.cpu()

    return {
        "u": u.resolve_conj().numpy(),
        "s": s.resolve_conj().numpy(),
        "v": v.resolve_conj().numpy()
    }

def tensorflow_svd_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        full_matrices = not input.get("some", True)
        
        # Apply tensorflow equivalent of svd
        s, u, v = tf.linalg.svd(input_tensor, full_matrices=full_matrices, compute_uv=True)

        if compute_uv := input.get("compute_uv", True):
            if cpu:
                u, s, v = u.numpy(), s.numpy(), v.numpy()
            return {
                "u": u,
                "s": s,
                "v": v
            }
        else:
            return {
                "s": s,
                "u": np.zeros([input_tensor.shape[0], input_tensor.shape[0]]).astype(np.float32),
                "v": np.zeros([input_tensor.shape[1], input_tensor.shape[1]]).astype(np.float32)
            }

def main():
    # Example input
    input_data = {
        "input": np.random.randn(5, 3).astype(np.float32),
        "some": True,
        "compute_uv": True
    }

    # Torch example
    torch_result = torch_svd_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_svd_version(input_data)
    tf_result = {key: value if isinstance(value, np.ndarray) else value.numpy() for key, value in tf_result.items()}
    print("TensorFlow result:", tf_result)

    # Check if results are equal
    try:
        assert np.allclose(torch_result['s'], tf_result['s']), "Singular values do not match."
        if 'u' in torch_result:
            assert np.allclose(torch_result['u'], tf_result['u']), "U matrices are not close."
        if 'v' in torch_result:
            assert np.allclose(torch_result['v'], tf_result['v']), "V matrices are not close."
        print("equal")
    except AssertionError:
        print("not equal")

if __name__ == "__main__":
    main()