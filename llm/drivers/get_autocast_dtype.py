import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.get_autocast_dtype("cuda" if not cpu else "cpu")

    if not cpu:
        pass

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])

        policy = tf.keras.mixed_precision.global_policy()
        if policy.compute_dtype == tf.float16:
            result = "float16"
        elif policy.compute_dtype == tf.bfloat16:
            result = "bfloat16"
        else:
            result = str(policy.compute_dtype).split("'")[1]


    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)

    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()