import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    bins = input.get("bins", 100)
    min_val = input.get("min", 0)
    max_val = input.get("max", 0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.histc
    histogram = torch.histc(input_tensor, bins=bins, min=min_val, max=max_val)

    if not cpu:
        histogram = histogram.cpu()

    return {"histogram": histogram.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        bins = input.get("bins", 100)
        min_val = input.get("min", 0)
        max_val = input.get("max", 0)

        # Logic to handle min=0 and max=0 as in torch.histc
        if min_val == 0 and max_val == 0:
            min_val = tf.reduce_min(input_tensor).numpy()
            max_val = tf.reduce_max(input_tensor).numpy()

        # Apply TensorFlow equivalent
        histogram = tf.histogram_fixed_width(input_tensor, [min_val, max_val], nbins=bins)

        return {"histogram": histogram.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1.0, 2.0, 1.0], dtype=np.float32),
        "bins": 4,
        "min": 0,
        "max": 3
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion and print
    if np.allclose(torch_result["histogram"], tf_result["histogram"], atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()