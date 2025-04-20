import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    padding = input["padding"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.nn.ReflectionPad2d
    reflection_pad = torch.nn.ReflectionPad2d(padding)
    output_tensor = reflection_pad(input_tensor)

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

        # TF Equivalent operation: tf.pad with 'REFLECT' mode
        if isinstance(padding, int):
            paddings = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
        else:
            paddings = [[0, 0], [0, 0], 
                        [padding[2], padding[3]], 
                        [padding[0], padding[1]]]

        output_tensor = tf.pad(input_tensor, paddings, mode='REFLECT')

    return {"output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3),
        "padding": 2,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and print the result
    assert np.allclose(torch_result["output"], tf_result["output"]), "Outputs are not equal"
    if np.allclose(torch_result["output"], tf_result["output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()