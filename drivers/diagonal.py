import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"]) 
    offset = input.get("offset", 0)
    dim1 = input.get("dim1", 0)
    dim2 = input.get("dim2", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.diagonal
    diagonal = torch.diagonal(input_tensor, offset=offset, dim1=dim1, dim2=dim2)

    if not cpu:
        diagonal = diagonal.cpu()

    return {"diagonal": diagonal.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    # Determine device
    device = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        offset = input.get("offset", 0)
        dim1 = input.get("dim1", 0)
        dim2 = input.get("dim2", 1)

        # Calculate diagonal
        diag = tf.linalg.diag_part(tf.transpose(input_tensor, perm=[dim1, dim2]))
        
        if offset != 0:
            if offset > 0:
                diag = diag[:, offset:]
            else:
                diag = diag[:, :offset]

        return {"diagonal": diag.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9], [0.7, 0.8, 0.4]], dtype=np.float32),
        "offset": 0,
        "dim1": 0,
        "dim2": 1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality
    assert np.array_equal(torch_result["diagonal"], tf_result["diagonal"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()