import torch
import tensorflow as tf
import numpy as np
from src.setseed import set_seed

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    offset = input.get("offset", 0)
    dim1 = input.get("dim1", -2)
    dim2 = input.get("dim2", -1)

    # Apply to torch.diag_embed
    result = torch.diag_embed(input_tensor, offset=offset, dim1=dim1, dim2=dim2)

    if not cpu:
        result = result.cpu()

    return {"diag_embed_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        offset = input.get("offset", 0)

        # Apply to TensorFlow equivalent (tf.linalg.diag with adjustments if necessary)
        if offset == 0:
            result = tf.linalg.diag(input_tensor)
        else:
            # For non-zero offsets, TensorFlow does not directly support what PyTorch's diag_embed supports (upper or lower triangular embeddings)
            raise NotImplementedError("Non-zero offsets are not directly supported by tf.linalg.diag equivalent in TensorFlow.")

        return {"diag_embed_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "offset": 0,
        "dim1": -2,
        "dim2": -1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    assert np.array_equal(torch_result["diag_embed_result"], tf_result["diag_embed_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()