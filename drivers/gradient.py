import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    spacing = input_dict.get("spacing", 1)
    dim = input_dict.get("dim", None)
    edge_order = input_dict.get("edge_order", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.gradient(input_tensor, spacing=spacing, dim=dim, edge_order=edge_order)
    
    if not cpu:
        if isinstance(result, tuple):
            result = tuple(r.cpu() for r in result)
        else:
            result = result.cpu()
            
    if isinstance(result, tuple):
        result_np = tuple(r.numpy() for r in result)
    else:
        result_np = result.numpy()

    return {"result": result_np}

def tensorflow_version(input_dict, cpu=True):
    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    spacing = input_dict.get("spacing", 1)
    dim = input_dict.get("dim", None)
    edge_order = input_dict.get("edge_order", 1)

    if isinstance(spacing, (int, float)):
        spacing = [spacing] * len(input_tensor.shape)
    elif isinstance(spacing, list):
        pass
    else:
        spacing = [tf.constant(s, dtype=tf.float32) for s in spacing]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        
        def compute_gradient(tensor, axis, spacing, edge_order):
            if axis is None:
                axis = list(range(len(tensor.shape)))
            elif not isinstance(axis, list):
                axis = [axis]

            grads = []
            for i in axis:
                if isinstance(spacing, list):
                    h = spacing[i]
                else:
                    h = 1.0

                if isinstance(h, tf.Tensor):
                    h = tf.cast(h, dtype=tf.float32)
                else:
                    h = tf.cast(h, dtype=tf.float32)
                    
                slices = [slice(None)] * len(tensor.shape)

                if edge_order == 1:
                    if tensor.shape[i] < 2:
                        grad = tf.zeros_like(tensor)
                        grads.append(grad)
                        continue
                    
                    grad_list = []

                    slices[i] = slice(1, None)
                    forward = tensor[tuple(slices)]

                    slices[i] = slice(0, -1)
                    backward = tensor[tuple(slices)]
                    
                    grad = (forward - backward) / h
                    
                    slices[i] = slice(2, None)
                    forward_internal = tensor[tuple(slices)]
                    slices[i] = slice(0, -2)
                    backward_internal = tensor[tuple(slices)]

                    grad_internal = (forward_internal - backward_internal) / (2*h)
                    
                    temp_grad = np.zeros(tensor.shape)
                    slices[i] = slice(1, -1)
                    temp_grad[tuple(slices)] = grad_internal.numpy()

                    slices[i] = 0
                    slices[i] = slice(0,1)
                    temp_grad[tuple(slices)] = ((tensor[tuple(slices[k] for k in range(i)] + [slice(1,2)] + [slices[k] for k in range(i+1,len(slices))])] - tensor[tuple(slices[k] for k in range(i)] + [slice(0,1)] + [slices[k] for k in range(i+1,len(slices))] )]) / h).numpy()
                    slices[i] = -1
                    slices[i] = slice(-1,None)
                    temp_grad[tuple(slices)] = ((tensor[tuple(slices[k] for k in range(i)] + [slice(-1,None)] + [slices[k] for k in range(i+1,len(slices))])] - tensor[tuple(slices[k] for k in range(i)] + [slice(-2,-1)] + [slices[k] for k in range(i+1,len(slices))] )]) / h).numpy()
                    
                    grad = tf.convert_to_tensor(temp_grad, dtype=tf.float32)
                    
                    grads.append(grad)
                else:
                    grad = tf.zeros_like(tensor)
                    grads.append(grad)


            return tuple(grads) if len(grads) > 1 else grads[0]
        
        grads = compute_gradient(input_tensor, dim, spacing, edge_order)

    if isinstance(grads, tuple):
        result_np = tuple(g.numpy() for g in grads)
    else:
        result_np = grads.numpy()

    return {"result": result_np}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 4, 8], [10, 20, 40, 80]], dtype=np.float32),
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if isinstance(torch_result["result"], tuple) and isinstance(tf_result["result"], tuple):
        for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
            assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"
    elif not isinstance(torch_result["result"], tuple) and not isinstance(tf_result["result"], tuple):
         assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    else:
        assert False, "Tuple type mismatch"

    input_data = {
        "input": np.array([[1, 2, 4, 8], [10, 20, 40, 80]], dtype=np.float32),
        "spacing": 2.0
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if isinstance(torch_result["result"], tuple) and isinstance(tf_result["result"], tuple):
        for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
            assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"
    elif not isinstance(torch_result["result"], tuple) and not isinstance(tf_result["result"], tuple):
         assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    else:
        assert False, "Tuple type mismatch"

    input_data = {
        "input": np.array([[1, 2, 4, 8], [10, 20, 40, 80]], dtype=np.float32),
        "dim": 1
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if isinstance(torch_result["result"], tuple) and isinstance(tf_result["result"], tuple):
        for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
            assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"
    elif not isinstance(torch_result["result"], tuple) and not isinstance(tf_result["result"], tuple):
         assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    else:
        assert False, "Tuple type mismatch"

    input_data = {
        "input": np.array([[1, 2, 4, 8], [10, 20, 40, 80]], dtype=np.float32),
        "spacing": [3.0, 2.0]
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if isinstance(torch_result["result"], tuple) and isinstance(tf_result["result"], tuple):
        for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
            assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"
    elif not isinstance(torch_result["result"], tuple) and not isinstance(tf_result["result"], tuple):
         assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    else:
        assert False, "Tuple type mismatch"

    coords = (np.array([0, 2]), np.array([0, 3, 6, 9]))
    input_data = {
        "input": np.array([[1, 2, 4, 8], [10, 20, 40, 80]], dtype=np.float32),
        "spacing": coords
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if isinstance(torch_result["result"], tuple) and isinstance(tf_result["result"], tuple):
        for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
            assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"
    elif not isinstance(torch_result["result"], tuple) and not isinstance(tf_result["result"], tuple):
         assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    else:
        assert False, "Tuple type mismatch"
        
    print("Success")

if __name__ == "__main__":
    main()