import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    if cpu:
        tensors = [torch.tensor(t) for t in input["tensors"]]
    else:
        tensors = [torch.tensor(t).cuda() for t in input["tensors"]]
    
    # Apply torch.dstack
    output_tensor = torch.dstack(tensors)
    
    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"dstack_result": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensors = [tf.constant(t) for t in input["tensors"]]
        
        # Ensure all tensors are at least 3D by using tf.expand_dims
        tensors = [tf.expand_dims(tf.expand_dims(t, axis=0), axis=-1) if len(t.shape) == 1 else tf.expand_dims(t, axis=-1) for t in tensors]
        
        # Apply tf.stack along axis 2
        output_tensor = tf.concat(tensors, axis=2)

    return {"dstack_result": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "tensors": [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        ]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion
    torch_output = torch_result["dstack_result"]
    tf_output = tf_result["dstack_result"]
    
    if np.allclose(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()