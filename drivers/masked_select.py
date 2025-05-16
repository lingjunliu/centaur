import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    device = torch.device("cpu" if cpu else "cuda")
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"]).to(device)
    mask_tensor = torch.tensor(input["mask"], dtype=torch.bool).to(device)
    
    # Apply to torch.masked_select
    output_tensor = torch.masked_select(input_tensor, mask_tensor)
    
    return {"masked_select_result": output_tensor.cpu().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        mask_tensor = tf.constant(input["mask"], dtype=tf.bool)

        # Apply to TensorFlow equivalent
        masked_indices = tf.where(mask_tensor)
        output_tensor = tf.gather_nd(input_tensor, masked_indices)

        return {"masked_select_result": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "mask":  np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert results are equal
    torch_output = torch_result["masked_select_result"]
    tf_output = tf_result["masked_select_result"]
    
    assert np.array_equal(torch_output, tf_output), 'not equal'
    print('equal')

if __name__ == "__main__":
    main()