import numpy as np

# Torch SiLU function
def torch_silu(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Apply the SiLU function
    silu = torch.nn.SiLU(inplace=input.get("inplace", False))
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        silu = silu.cuda()
    
    output_tensor = silu(input_tensor)
    
    if not cpu:
        output_tensor = output_tensor.cpu()
    
    return {"SiLU_output": output_tensor.numpy()}

# TensorFlow equivalent SiLU function
def tf_silu(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        
        # Apply the SiLU function
        output_tensor = input_tensor * tf.nn.sigmoid(input_tensor)
        
        return {"SiLU_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3).astype(np.float32),  # Random input example
        "inplace": False   # Note: TensorFlow doesn't have an "inplace" option.
    }

    # Torch SiLU example
    torch_result = torch_silu(input_data)
    print("Torch result:", torch_result)

    # TensorFlow SiLU example
    tf_result = tf_silu(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.allclose(torch_result["SiLU_output"], tf_result["SiLU_output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()