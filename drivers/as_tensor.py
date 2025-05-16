import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    data = input["data"]
    dtype = torch.float32 if input.get("dtype") == 'float32' else None
    device = torch.device('cpu') if cpu else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Apply to torch.as_tensor
    tensor = torch.as_tensor(data, dtype=dtype, device=device)
    
    if cpu and device != torch.device('cpu'):
        tensor = tensor.cpu()

    return {"as_tensor_result": tensor.cpu().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary
        data = input["data"]
        dtype = tf.float32 if input.get("dtype") == 'float32' else None

        # Apply to tf.convert_to_tensor
        tensor = tf.convert_to_tensor(data, dtype=dtype)

        return {"as_tensor_result": tensor.numpy()}

def main():
    # Example input
    input_data = {
        "data": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dtype": 'float32'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Check for equality
    torch_output = torch_result["as_tensor_result"]
    tf_output = tf_result["as_tensor_result"]

    if np.array_equal(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()