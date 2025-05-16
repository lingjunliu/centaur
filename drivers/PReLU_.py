import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    num_parameters = input.get("num_parameters", 1)
    init = input.get("init", 0.25)
    
    # Initialize PReLU
    prelu = torch.nn.PReLU(num_parameters=num_parameters, init=init)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        prelu = prelu.cuda()
    
    # Apply PReLU
    output = prelu(input_tensor)
    
    if not cpu:
        output = output.cpu()

    return {"output": output.detach().numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        num_parameters = input.get("num_parameters", 1)
        init = input.get("init", 0.25)

        # Initialize learnable alpha
        alphas = tf.Variable(np.full((num_parameters,), init))
        
        # Apply PReLU
        output = tf.maximum(0.0, input_tensor) + alphas * tf.minimum(0.0, input_tensor)
        
        return {"output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3).astype(np.float32),
        "num_parameters": 3,
        "init": 0.1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = torch_result["output"]
    tf_output = tf_result["output"]

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()