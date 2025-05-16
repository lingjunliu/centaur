import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    vital = input_dict["vital"]
    vital_admit = input_dict.get("vital_admit", 0.5)
    vital_severe = input_dict.get("vital_severe", 0.2)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.where(input_tensor >= vital, torch.ones_like(input_tensor),
                           torch.where(input_tensor >= vital * vital_admit, torch.full_like(input_tensor, 0.5),
                                    torch.where(input_tensor >= vital * vital_severe, torch.full_like(input_tensor, 0.2), torch.zeros_like(input_tensor))))


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
        vital = input_dict["vital"]
        vital_admit = input_dict.get("vital_admit", 0.5)
        vital_severe = input_dict.get("vital_severe", 0.2)

        vital_admit = tf.constant(vital_admit, dtype=input_tensor.dtype)
        vital_severe = tf.constant(vital_severe, dtype=input_tensor.dtype)
        
        result = tf.where(input_tensor >= vital, tf.ones_like(input_tensor),
                           tf.where(input_tensor >= vital * vital_admit, tf.fill(input_tensor.shape, 0.5),
                                    tf.where(input_tensor >= vital * vital_severe, tf.fill(input_tensor.shape, 0.2), tf.zeros_like(input_tensor))))
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.6, 0.3, 0.1, 0.8], dtype=np.float32),
        "vital": 0.5,
        "vital_admit": 0.5,
        "vital_severe": 0.2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()