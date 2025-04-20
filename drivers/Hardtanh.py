import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input['input'])
    min_val = input.get('min_val', -1.0)
    max_val = input.get('max_val', 1.0)
    inplace = input.get('inplace', False)
    
    # Apply torch.nn.Hardtanh
    hardtanh = torch.nn.Hardtanh(min_val, max_val, inplace)
    result = hardtanh(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"hardtanh_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input['input'])
        min_val = input.get('min_val', -1.0)
        max_val = input.get('max_val', 1.0)
        
        # Apply TensorFlow equivalent
        result = tf.clip_by_value(input_tensor, min_val, max_val)

        return {"hardtanh_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 1.8], [-2.0, 0.6, 0.9]], dtype=np.float32),
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion for comparisons
    assert np.allclose(torch_result["hardtanh_result"], tf_result["hardtanh_result"]), "Results are not equal!"
    if np.allclose(torch_result["hardtanh_result"], tf_result["hardtanh_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()