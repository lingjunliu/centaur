import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    try:
        def traced_fn(x):
            return x + 1

        script = torch.jit.script(traced_fn)
        result = "Success"
    except Exception as e:
        result = str(e)
    
    if not cpu and isinstance(result, torch.Tensor):
        result = result.cpu()
    
    return {"result": str(result)}

def tensorflow_version(input_dict, cpu=True):

    @tf.function
    def my_func(x):
        return x + 1

    try:
        concrete_function = my_func.get_concrete_function(tf.constant(input_dict["input"]))
        result = "Function invoked by tracing"
    except Exception as e:
        result = str(e)

    return {"result": str(result)}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert "TracerWarning" in torch_result["result"] or "Success" in torch_result["result"]
    assert "Function invoked by tracing" in tf_result["result"]
    print("Success")

if __name__ == "__main__":
    main()