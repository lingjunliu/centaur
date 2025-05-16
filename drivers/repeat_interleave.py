import numpy as np

# Function to ensure reproducibility
def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input["input"])
    repeats = input["repeats"]
    dim = input.get("dim", None)
    
    result = torch.repeat_interleave(input_tensor, repeats=repeats, dim=dim)
    
    if not cpu:
        result = result.cpu()
        
    return {"repeat_interleave_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    device_string = "/cpu:0" if cpu else "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        repeats = input["repeats"]
        dim = input.get("dim", None)
        
        if dim is None:
            result = tf.repeat(input_tensor, repeats)
        else:
            result = tf.repeat(input_tensor, repeats, axis=dim)
        
        result_np = result.numpy()
        
        return {"repeat_interleave_result": result_np}

def main():
    # Example input for first case example: torch.repeat_interleave(input, repeats, dim=None)
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "repeats": 2,
        "dim": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result for case 1:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result for case 1:", tf_result)

    # Check results
    if np.array_equal(torch_result["repeat_interleave_result"], tf_result["repeat_interleave_result"]):
        print("equal")
    else:
        print("not equal")
    
    # Second example input for case example: torch.repeat_interleave(repeats)
    input_data = {
        "input": np.array([0.5, 0.3, 0.8], dtype=np.float32),
        "repeats": [2, 1, 3],
        "dim": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result for case 2:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result for case 2:", tf_result)

    # Check results
    if np.array_equal(torch_result["repeat_interleave_result"], tf_result["repeat_interleave_result"]):
        print("equal")
    else:
        print("not equal")


if __name__ == "__main__":
    main()