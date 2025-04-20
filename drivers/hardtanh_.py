import numpy as np

def torch_hardtanh(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    min_val = input.get("min_val", -1.0)
    max_val = input.get("max_val", 1.0)
    inplace = input.get("inplace", False)

    # Apply PyTorch hardtanh
    result = torch.nn.functional.hardtanh(input_tensor, min_val=min_val, max_val=max_val, inplace=inplace)

    if not cpu:
        result = result.cpu()

    return {"hardtanh_result": result.numpy()}

def tensorflow_hardtanh(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        min_val = input.get("min_val", -1.0)
        max_val = input.get("max_val", 1.0)

        # Apply TensorFlow equivalent
        result = tf.clip_by_value(input_tensor, clip_value_min=min_val, clip_value_max=max_val)

        return {"hardtanh_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-2.0, 0.0, 1.5], [3.0, -1.0, 0.5]], dtype=np.float32),
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False
    }

    # Torch example
    torch_result = torch_hardtanh(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_hardtanh(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = torch_result["hardtanh_result"]
    tf_output = tf_result["hardtanh_result"]

    if np.allclose(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()