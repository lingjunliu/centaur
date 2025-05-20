def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    # Unpack inputs from dictionary
    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict["dim"]
    index = torch.tensor(input_dict["index"])
    src = torch.tensor(input_dict["src"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()
        src = src.cuda()
    
    # Perform torch addition
    result = torch.scatter(input_tensor, dim, index, src)
    
    # Move result to CPU for consistent return format
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict):
    import tensorflow as tf
    input_tensor = tf.constant(input_dict["input"])
    dim = input_dict["dim"]
    index = tf.constant(input_dict["index"])
    src = tf.constant(input_dict["src"])

    input_shape = tf.shape(input_tensor)
    batch_dims = tf.meshgrid(*[tf.range(s) for s in input_shape], indexing='ij')

    if dim == 0:
        scatter_indices = tf.stack([index, batch_dims[1]], axis=-1)
    elif dim == 1:
        scatter_indices = tf.stack([batch_dims[0], index], axis=-1)
    else:
        raise ValueError("Only dim 0 or 1 supported for 2D tensors.")

    scatter_indices = tf.cast(scatter_indices, tf.int64)
    result = tf.tensor_scatter_nd_update(input_tensor, scatter_indices, src)

    return {"result": result.numpy()}

def main():
    import numpy as np

    # Sample 2D input example
    input_dict = {
        "input": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "dim": 1,
        "index": [[0, 1, 2], [2, 0, 1]],
        "src": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    }

    torch_output = torch_version(input_dict)
    tf_output = tensorflow_version(input_dict)

    np.testing.assert_allclose(torch_output["result"], tf_output["result"], atol=0.01)
    print("Outputs match within tolerance.")

if __name__ == "__main__":
    main()
