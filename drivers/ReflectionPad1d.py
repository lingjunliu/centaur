import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Retrieve input data and padding parameter
    input_tensor = torch.tensor(input["input"])
    padding = input["padding"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply ReflectionPad1d
    reflection_pad = torch.nn.ReflectionPad1d(padding)
    output_tensor = reflection_pad(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return { 'output': output_tensor.numpy() }

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Retrieve input data and padding parameter
        input_tensor = tf.constant(input["input"])
        padding = input["padding"]
        
        if isinstance(padding, int):
            padding = [[0, 0], [0, 0], [padding, padding]]
        elif isinstance(padding, tuple):
            padding = [[0, 0], [0, 0], list(padding)]

        # Apply reflection padding using tf.pad
        output_tensor = tf.pad(input_tensor, paddings=padding, mode='REFLECT')
        
        return { 'output': output_tensor.numpy() }

def main():
    # Example input
    input_data = {
        "input": np.arange(8, dtype=np.float32).reshape(1, 2, 4),  # Shape (1, 2, 4)
        "padding": 2,  # Equivalent to torch.nn.ReflectionPad1d(2)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparison
    if np.array_equal(torch_result['output'], tf_result['output']):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()