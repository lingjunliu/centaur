import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    alpha = input.get("alpha", 1.0)
    inplace = input.get("inplace", False)
    input_tensor = torch.tensor(input["input"])

    # Apply to torch.nn.ELU
    elu = torch.nn.ELU(alpha=alpha, inplace=inplace)
    if not cpu:
        input_tensor = input_tensor.to("cuda")
        elu.to("cuda")
        
    output_tensor = elu(input_tensor)
    
    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        alpha = input.get("alpha", 1.0)
        input_tensor = tf.constant(input["input"])

        # Apply to TensorFlow equivalent
        output_tensor = tf.nn.elu(input_tensor) * alpha

        return {"output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [-0.2, 0.6, -0.9]], dtype=np.float32),
        "alpha": 1.0,
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    # As each framework handles floating-point math in slightly different ways,
    # a tolerance value can be adjusted for comparison.
    if np.allclose(torch_result["output"], tf_result["output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()