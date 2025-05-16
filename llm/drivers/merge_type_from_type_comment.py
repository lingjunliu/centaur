import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    type_comment = input_dict["type_comment"]

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    if type_comment == "torch.float32":
        result = input_tensor.float()
    elif type_comment == "torch.float64":
        result = input_tensor.double()
    elif type_comment == "torch.int32":
        result = input_tensor.int()
    elif type_comment == "torch.int64":
        result = input_tensor.long()
    elif type_comment == "torch.float16":
        result = input_tensor.half()
    elif type_comment == "torch.bfloat16":
        result = input_tensor.bfloat16()
    elif type_comment == "torch.int8":
        result = input_tensor.char()
    elif type_comment == "torch.uint8":
        result = input_tensor.byte()
    elif type_comment == "torch.int16":
        result = input_tensor.short()
    elif type_comment == "torch.bool":
        result = input_tensor.bool()
    else:
        raise ValueError(f"Unsupported type comment: {type_comment}")
    
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
        type_comment = input_dict["type_comment"]

        if type_comment == "torch.float32":
            dtype = tf.float32
        elif type_comment == "torch.float64":
            dtype = tf.float64
        elif type_comment == "torch.int32":
            dtype = tf.int32
        elif type_comment == "torch.int64":
            dtype = tf.int64
        elif type_comment == "torch.float16":
            dtype = tf.float16
        elif type_comment == "torch.bfloat16":
            dtype = tf.bfloat16
        elif type_comment == "torch.int8":
            dtype = tf.int8
        elif type_comment == "torch.uint8":
            dtype = tf.uint8
        elif type_comment == "torch.int16":
            dtype = tf.int16
        elif type_comment == "torch.bool":
            dtype = tf.bool
        else:
            raise ValueError(f"Unsupported type comment: {type_comment}")

        result = tf.cast(input_tensor, dtype)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int64),
        "type_comment": "torch.float32"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "type_comment": "torch.int64"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([True, False, True], dtype=bool),
        "type_comment": "torch.int64"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()