import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    pred = input_dict["pred"]
    true_fn = input_dict["true_fn"]
    false_fn = input_dict["false_fn"]
    operands = input_dict.get("operands", ())

    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)

    if not cpu:
        if isinstance(pred, torch.Tensor):
            pred = pred.cuda()
        operands_cuda = []
        for operand in operands:
            if isinstance(operand, np.ndarray):
                operands_cuda.append(torch.tensor(operand).cuda())
            elif isinstance(operand, torch.Tensor):
                operands_cuda.append(operand.cuda())
            else:
                operands_cuda.append(operand)
        operands = tuple(operands_cuda)
    else:
        operands_torch = []
        for operand in operands:
            if isinstance(operand, np.ndarray):
                operands_torch.append(torch.tensor(operand))
            else:
                operands_torch.append(operand)
        operands = tuple(operands_torch)

    def true_fn_wrapper(*args):
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in args]
        if not cpu:
            tensors = [x.cuda() if isinstance(x, torch.Tensor) else x for x in tensors]
        result = true_fn(*tensors)
        if not cpu and isinstance(result, torch.Tensor):
            result = result.cpu()
        return result

    def false_fn_wrapper(*args):
        tensors = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in args]
        if not cpu:
            tensors = [x.cuda() if isinstance(x, torch.Tensor) else x for x in tensors]
        result = false_fn(*tensors)
        if not cpu and isinstance(result, torch.Tensor):
            result = result.cpu()
        return result

    result = torch.cond(pred, true_fn_wrapper, false_fn_wrapper, operands=operands)
    
    if isinstance(result, torch.Tensor):
        if not cpu:
            result = result.cpu()
        result = result.numpy()
    elif isinstance(result, tuple):
        result = tuple(r.numpy() if isinstance(r, torch.Tensor) else r for r in result)
    elif isinstance(result, list):
        result = [r.numpy() if isinstance(r, torch.Tensor) else r for r in result]

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    pred = input_dict["pred"]
    true_fn = input_dict["true_fn"]
    false_fn = input_dict["false_fn"]
    operands = input_dict.get("operands", ())
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):

        if isinstance(pred, np.ndarray):
            pred = tf.constant(pred)
        
        operands_tf = []
        for operand in operands:
            if isinstance(operand, np.ndarray):
                operands_tf.append(tf.constant(operand))
            else:
                operands_tf.append(operand)
        operands_tf = tuple(operands_tf)
        
        def true_fn_wrapper(*args):
            result = true_fn(*args)
            if isinstance(result, tf.Tensor):
                result = result.numpy()
            return result
        
        def false_fn_wrapper(*args):
            result = false_fn(*args)
            if isinstance(result, tf.Tensor):
                result = result.numpy()
            return result
        
        if isinstance(pred, tf.Tensor):
            pred_value = pred.numpy()
        else:
            pred_value = pred
            
        if pred_value:
            result = true_fn_wrapper(*operands_tf)
        else:
            result = false_fn_wrapper(*operands_tf)

        if isinstance(result, np.ndarray):
            pass
        elif isinstance(result, tuple):
             result = tuple(r.numpy() if isinstance(r, tf.Tensor) else r for r in result)
        elif isinstance(result, list):
            result = [r.numpy() if isinstance(r, tf.Tensor) else r for r in result]

    return {"result": result}

def main():
    A_TOL = 0.01
    
    def true_fn(x):
        return x + 1
    
    def false_fn(x):
        return x - 1

    input_data = {
        "pred": np.array(True),
        "true_fn": true_fn,
        "false_fn": false_fn,
        "operands": (np.array(5),)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    def true_fn_nested(x, y):
        return (x + 1, y * 2)

    def false_fn_nested(x, y):
        return (x - 1, y / 2)

    input_data_nested = {
        "pred": False,
        "true_fn": true_fn_nested,
        "false_fn": false_fn_nested,
        "operands": (np.array(5), np.array(10))
    }

    torch_result_nested = torch_version(input_data_nested)
    tf_result_nested = tensorflow_version(input_data_nested)

    assert np.allclose(torch_result_nested["result"][0], tf_result_nested["result"][0], atol=A_TOL), "Nested results do not match"
    assert np.allclose(torch_result_nested["result"][1], tf_result_nested["result"][1], atol=A_TOL), "Nested results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()