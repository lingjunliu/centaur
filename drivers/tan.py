import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    device = torch.device('cpu' if cpu else 'cuda')
    input_tensor = input_tensor.to(device)

    # Apply torch.tan
    result_tensor = torch.tan(input_tensor)

    return {"result": result_tensor.cpu().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent
        result_tensor = tf.math.tan(input_tensor)

        return {"result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([-1.2027, -1.7687, 0.4412, -1.3856], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality
    assert np.allclose(torch_result["result"], tf_result["result"], atol=1e-5), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()