import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    threshold = input["threshold"]
    value = input["value"]
    inplace = input["inplace"]

    # Apply the threshold
    threshold_layer = torch.nn.Threshold(threshold, value, inplace=inplace)
    output_tensor = threshold_layer(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"threshold_output": output_tensor.numpy()}

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
        threshold = input["threshold"]
        value = input["value"]

        # Apply the threshold
        output_tensor = tf.where(input_tensor > threshold, input_tensor, value)
        
        return {"threshold_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.05, 0.8], [0.2, 0.6, 0.09]], dtype=np.float32),
        "threshold": 0.1,
        "value": 20,
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparison
    if np.allclose(torch_result["threshold_output"], tf_result["threshold_output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()