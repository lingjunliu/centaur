import numpy as np

def torch_version(input, cpu=True):
    import torch
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.numel
    num_elements = torch.numel(input_tensor)

    return {"num_elements": int(num_elements)}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.convert_to_tensor(input["input"])

        # Apply to TensorFlow equivalent
        num_elements = tf.size(input_tensor)

        return {"num_elements": int(num_elements.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(4, 4).astype(np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    if torch_result["num_elements"] == tf_result["num_elements"]:
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()