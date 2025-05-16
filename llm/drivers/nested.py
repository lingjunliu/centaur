import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_nested_tensor = input_dict["input"]
    
    if not cpu:
        input_nested_tensor = input_nested_tensor.cuda()

    result = input_nested_tensor.to_padded_tensor(padding=input_dict.get("padding", 0.0))
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    input_nested_tensor = input_dict["input"]
    padding_value = input_dict.get("padding", 0.0)

    ragged_tensor = tf.ragged.constant([x.numpy() for x in input_nested_tensor])

    result = ragged_tensor.to_tensor(default_value=padding_value).numpy()

    return {"result": result}

def main():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    A_TOL = 0.01

    nested_list = [torch.randn(3), torch.randn(4), torch.randn(2)]
    input_data = {
        "input": nested_list
    }

    torch_nested_tensor = torch.nested.nested_tensor(input_data["input"])
    input_data["input"] = torch_nested_tensor

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()