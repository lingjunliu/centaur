import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    obj = input_dict["obj"]
    example_inputs = input_dict.get("example_inputs", None)

    if not cpu:
        if isinstance(obj, torch.Tensor):
            obj = obj.cuda()
        elif isinstance(obj, dict):
            obj = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
        elif isinstance(obj, list):
            obj = [v.cuda() if isinstance(v, torch.Tensor) else v for v in obj]
        if example_inputs is not None:
            if isinstance(example_inputs, list):
                example_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs]
            elif isinstance(example_inputs, dict):
                example_inputs = {k: [x.cuda() if isinstance(x, torch.Tensor) else x for x in v] for k, v in example_inputs.items()}
    
    if not isinstance(obj, type):
        def foo(x, y):
            if torch.max(x) > torch.max(y):
                r = x
            else:
                r = y
            return r

        scripted_obj = torch.jit.script(foo)
        input_tensor = torch.ones(2, 2)
        other_tensor = torch.ones(2, 2)
        if not cpu:
            input_tensor = input_tensor.cuda()
            other_tensor = other_tensor.cuda()
        result = scripted_obj(input_tensor, other_tensor)
    else:
        scripted_obj = torch.jit.script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=example_inputs)
        if hasattr(scripted_obj, 'forward'):
            dummy_input = torch.randn(2, 3)
            if not cpu:
                dummy_input = dummy_input.cuda()
            result = scripted_obj(dummy_input)
        else:
            result = scripted_obj(torch.ones(2, 2), torch.ones(2, 2))


    if not cpu:
        if isinstance(result, torch.Tensor):
            result = result.cpu()
        elif isinstance(result, dict):
            result = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        elif isinstance(result, list):
            result = [v.cpu() if isinstance(v, torch.Tensor) else v for v in result]

    if isinstance(result, torch.Tensor):
        return {"result": result.numpy()}
    else:
        return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        obj = input_dict["obj"]
        example_inputs = input_dict.get("example_inputs", None)
        
        def eager_func(x, y):
            return tf.cond(tf.math.reduce_max(x) > tf.math.reduce_max(y), lambda: x, lambda: y)

        input_tensor = tf.ones((2, 2))
        other_tensor = tf.ones((2, 2))

        result = eager_func(input_tensor, other_tensor)

    if isinstance(result, tf.Tensor):
        return {"result": result.numpy()}
    elif isinstance(result, dict):
        return {"result": {k: v.numpy() if isinstance(v, tf.Tensor) else v for k, v in result.items()}}
    elif isinstance(result, list):
        return {"result": [v.numpy() if isinstance(v, tf.Tensor) else v for v in result]}
    else:
        return {"result": result}

def main():
    A_TOL = 0.01
    def func(x, y):
        return tf.where(tf.reduce_max(x) > tf.reduce_max(y), x, y)
    input_data = {
        "obj": func
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()