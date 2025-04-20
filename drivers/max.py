import numpy as np

# Define equivalent for tensorflow max along specified dimensions
def tf_max(input, axis):
    import tensorflow as tf
    values = tf.reduce_max(input, axis=axis)
    indices = tf.argmax(input, axis=axis)
    return values, indices

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if "dim" not in input:
        output = torch.max(input_tensor)
    else:
        dim = input["dim"]
        keepdim = input.get("keepdim", False)
        output = torch.max(input_tensor, dim=dim, keepdim=keepdim)

    if not cpu:
        if isinstance(output, (tuple, list)):
            output = tuple(out.cpu() for out in output)
        else:
            output = output.cpu()

    if "dim" not in input:
        return {"max": float(output.item())}
    else:
        values, indices = output
        return {"values": values.numpy(), "indices": indices.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input["input"])
        
        if "dim" not in input:
            output = tf.reduce_max(input_tensor)
        
        else:
            axis = input["dim"]
            output = tf_max(input_tensor, axis)        

        if "dim" not in input:
            return {"max": float(output.numpy())}
        else:
            values, indices = output
            return {"values": values.numpy(), "indices": indices.numpy()}        

def main():
    # Example input for case 1
    input_data_case_1 = {
        "input": np.array([0.5, 0.3, 0.8], dtype=np.float32)
    }

    # Example for case 1
    torch_result_1 = torch_version(input_data_case_1)
    tf_result_1 = tensorflow_version(input_data_case_1)
    assert torch_result_1["max"] == tf_result_1["max"], "Case 1: Results differ!"
    print("Case 1: equal")

    # Example input for case 2
    input_data_case_2 = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1,
        "keepdim": False
    }

    # Example for case 2
    torch_result_2 = torch_version(input_data_case_2)
    tf_result_2 = tensorflow_version(input_data_case_2)
    np.testing.assert_array_equal(torch_result_2["values"], tf_result_2["values"])
    np.testing.assert_array_equal(torch_result_2["indices"], tf_result_2["indices"])
    print("Case 2: equal")

if __name__ == "__main__":
    main()