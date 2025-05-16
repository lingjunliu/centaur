import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    try:
        @torch.jit.script
        def my_func(x):
            return x + 1
        
        input_tensor = torch.tensor(input_dict["input"])
        result = my_func(input_tensor)
    except RuntimeError as e:
        result = str(e)
        return {"result": result}
    else:
        return {"result": "No TracingCheckError"}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        try:
            @tf.function
            def my_func(x):
                return x + 1
            
            result = my_func(input_tensor)
        except Exception as e:
            result = str(e)
            return {"result": result}
        else:
            return {"result": "No TracingCheckError"}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert torch_result["result"] == tf_result["result"], "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()