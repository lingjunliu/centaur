import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    start_dim = input.get("start_dim", 0)
    end_dim = input.get("end_dim", -1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.flatten
    output_tensor = torch.flatten(input_tensor, start_dim=start_dim, end_dim=end_dim)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"flattened": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        start_dim = input.get("start_dim", 0)
        end_dim = input.get("end_dim", -1)

        # Apply to TensorFlow equivalent (tf.reshape)
        original_shape = tf.shape(input_tensor)
        if end_dim == -1:
            end_dim = tf.size(original_shape) - 1
        
        flattened_size = tf.reduce_prod(original_shape[start_dim:end_dim+1])
        pre_shape = original_shape[:start_dim]
        post_shape = original_shape[end_dim+1:]
        new_shape = tf.concat([pre_shape, [flattened_size], post_shape], axis=0)
        
        output_tensor = tf.reshape(input_tensor, new_shape)

        return {"flattened": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
        "start_dim": 1,
        "end_dim": 2
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    torch_flattened = torch_result["flattened"]
    tf_flattened = tf_result["flattened"]

    if np.array_equal(torch_flattened, tf_flattened):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()