# TODO: Fix tf version
import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    hidden_tensor = torch.tensor(input["hidden"])
    
    if "input_size" in input.keys():
        input_size = input["input_size"]
    else:
        input_size = input_tensor.shape[-1]
    
    if "hidden_size" in input.keys():    
        hidden_size = input["hidden_size"]
    else:
        hidden_size = hidden_tensor.shape[-1]

    bias = input.get("bias", True)

    # Create GRUCell
    gru_cell = torch.nn.GRUCell(input_size, hidden_size, bias=bias)

    if not cpu:
        input_tensor = input_tensor.cuda()
        hidden_tensor = hidden_tensor.cuda()
        gru_cell = gru_cell.cuda()

    # Apply GRUCell
    output_tensor = gru_cell(input_tensor, hidden_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"output": output_tensor.detach().numpy()}

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
        hidden_tensor = tf.constant(input["hidden"])
        
        if "input_size" in input.keys():
            input_size = input["input_size"]
        else:
            input_size = input_tensor.shape[-1]
        
        if "hidden_size" in input.keys():    
            hidden_size = input["hidden_size"]
        else:
            hidden_size = hidden_tensor.shape[-1]

        bias = input.get("bias", True)

        # Create GRUCell
        gru_cell = tf.keras.layers.GRUCell(units=hidden_size, use_bias=bias)

        # Apply GRUCell
        output_tensor, _ = gru_cell(input_tensor, [hidden_tensor])

        return {"output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8]], dtype=np.float32),  # (batch_size, input_size)
        "hidden": np.array([[0.2, 0.6, 0.9]], dtype=np.float32),  # (batch_size, hidden_size)
        "input_size": 3,
        "hidden_size": 3,
        "bias": True
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["output"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["output"])

    # Assert that results are equal
    if np.allclose(torch_result["output"], tf_result["output"], atol=1e-2):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()
