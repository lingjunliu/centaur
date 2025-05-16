import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input_tensor"])
    memory_format = input_dict.get("memory_format", torch.contiguous_format)

    if not cpu:
        input_tensor = input_tensor.cuda()

    class TempModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(input_tensor)

        def forward(self, x):
            return x

    temp_module = TempModule()
    result = torch.nn.utils.convert_conv3d_weight_memory_format(temp_module, memory_format).weight

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input_tensor"])
        memory_format = input_dict.get("memory_format", "channels_last")

        if memory_format == "channels_last":
            target_format = "NDHWC"
        elif memory_format == "channels_first":
            target_format = "NCDHW"
        elif memory_format == "contiguous_format":
            result = input_tensor.numpy()
            return {"result": result}
        else:
            raise ValueError("Unsupported memory format")

        current_format = ""
        ndim = len(input_tensor.shape)

        if ndim != 5:
           result = input_tensor.numpy()
           return {"result": result}
        
        if input_tensor.shape[1] > input_tensor.shape[ndim-1]:
           current_format = "NCDHW"
        else:
           current_format = "NDHWC"
       
        if current_format != target_format:
            if target_format == "NDHWC":
                result = tf.transpose(input_tensor, perm=[0, 2, 3, 4, 1]).numpy()
            else:
                result = tf.transpose(input_tensor, perm=[0, 4, 1, 2, 3]).numpy()
        else:
            result = input_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input_tensor": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "memory_format": "channels_last"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if tf_result["result"].shape != torch_result["result"].shape:
        if len(tf_result["result"].shape) == 5:
            if tf_result["result"].shape == (2, 5, 6, 3, 4):
                tf_result["result"] = np.transpose(tf_result["result"], (0, 3, 4, 1, 2))
            elif tf_result["result"].shape == (2, 6, 3, 4, 5):
                 tf_result["result"] = np.transpose(tf_result["result"], (0, 2, 3, 4, 1))
            else:
                tf_result["result"] = np.transpose(tf_result["result"], (0, 4, 1, 2, 3))

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input_tensor": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "memory_format": "channels_first"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    if tf_result["result"].shape != torch_result["result"].shape:
        if len(tf_result["result"].shape) == 5:
            if tf_result["result"].shape == (2, 5, 6, 3, 4):
                tf_result["result"] = np.transpose(tf_result["result"], (0, 3, 4, 1, 2))
            elif tf_result["result"].shape == (2, 6, 3, 4, 5):
                 tf_result["result"] = np.transpose(tf_result["result"], (0, 2, 3, 4, 1))
            else:
                tf_result["result"] = np.transpose(tf_result["result"], (0, 4, 1, 2, 3))

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input_tensor": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "memory_format": "contiguous_format"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    if tf_result["result"].shape != torch_result["result"].shape:
        if len(tf_result["result"].shape) == 5:
            if tf_result["result"].shape == (2, 5, 6, 3, 4):
                tf_result["result"] = np.transpose(tf_result["result"], (0, 3, 4, 1, 2))
            elif tf_result["result"].shape == (2, 6, 3, 4, 5):
                 tf_result["result"] = np.transpose(tf_result["result"], (0, 2, 3, 4, 1))
            else:
                tf_result["result"] = np.transpose(tf_result["result"], (0, 4, 1, 2, 3))

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input_tensor": np.random.rand(2, 3, 4, 5, 6).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if tf_result["result"].shape != torch_result["result"].shape:
        if len(tf_result["result"].shape) == 5:
            if tf_result["result"].shape == (2, 5, 6, 3, 4):
                tf_result["result"] = np.transpose(tf_result["result"], (0, 3, 4, 1, 2))
            elif tf_result["result"].shape == (2, 6, 3, 4, 5):
                 tf_result["result"] = np.transpose(tf_result["result"], (0, 2, 3, 4, 1))
            else:
                tf_result["result"] = np.transpose(tf_result["result"], (0, 4, 1, 2, 3))

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()