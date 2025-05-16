import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    a = torch.tensor(input["a"])
    b = torch.tensor(input["b"])
    dims = input.get("dims", 2)
    
    # Ensure we are using the right device
    if not cpu:
        a = a.cuda()
        b = b.cuda()

    # Apply torch.tensordot
    result = torch.tensordot(a, b, dims=dims)

    if not cpu:
        result = result.cpu()

    return {"tensordot_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    device_string = "/cpu:0" if cpu else "/gpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary
        a = tf.constant(input["a"])
        b = tf.constant(input["b"])
        dims = input.get("dims", 2)
        
        # Apply tf.tensordot
        result = tf.tensordot(a, b, axes=dims)

        return {"tensordot_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "a": np.random.randn(3, 4, 5).astype(np.float32),
        "b": np.random.randn(4, 3, 2).astype(np.float32),
        "dims": ([1, 0], [0, 1])
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert results to check equality
    torch_res, tf_res = torch_result["tensordot_result"], tf_result["tensordot_result"]
    if np.allclose(torch_res, tf_res, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()