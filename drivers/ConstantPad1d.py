import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    padding = input["padding"]
    value = input["value"]

    # Apply torch.nn.ConstantPad1d
    pad = torch.nn.ConstantPad1d(padding, value)
    if not cpu:
        input_tensor = input_tensor.to('cuda')
        pad = pad.to('cuda')

    output_tensor = pad(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        padding = input["padding"]
        value = input["value"]

        # Compute padding
        if isinstance(padding, int):
            padding = (padding, padding)
        
        paddings = [[0, 0], [0, 0], [padding[0], padding[1]]]
        output_tensor = tf.pad(input_tensor, paddings, "CONSTANT", constant_values=value)

    return {"output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(1, 2, 4).astype(np.float32),
        "padding": 2,
        "value": 3.5
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_output = np.array(torch_result["output"])
    tf_output = np.array(tf_result["output"])

    assert np.allclose(torch_output, tf_output), "Output mismatch!"
    print("equal")

if __name__ == "__main__":
    main()