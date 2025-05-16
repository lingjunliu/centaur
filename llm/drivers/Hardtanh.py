import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    min_val = input_dict.get("min_val", -1.0)
    max_val = input_dict.get("max_val", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    hardtanh = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)
    
    if not cpu:
      hardtanh = hardtanh.cuda()
    
    result = hardtanh(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        min_val = input_dict.get("min_val", -1.0)
        max_val = input_dict.get("max_val", 1.0)

        result = tf.clip_by_value(input_tensor, clip_value_min=min_val, clip_value_max=max_val)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-2, -1, 0, 1, 2], dtype=np.float32),
        "min_val": -1.0,
        "max_val": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()