import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

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
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        spacing = input_dict.get("spacing", 1)
        dim = input_dict.get("dim", None)
        edge_order = input_dict.get("edge_order", 1)

        if isinstance(spacing, (int, float)):
            h = spacing
        elif isinstance(spacing, list) and all(isinstance(x, (int, float)) for x in spacing):
            h = spacing
        elif isinstance(spacing, tuple) and all(isinstance(x, (int, float)) for x in spacing):
            h = list(spacing)
        else:
            h = 1

        if dim is None:
            dim = list(range(len(input_tensor.shape)))
        elif isinstance(dim, int):
            dim = [dim]

        grads = []
        for i in dim:
            if isinstance(h, list):
                hi = h[i]
            else:
                hi = h

            if len(input_tensor.shape) == 1:
                if input_tensor.shape[0] < 2:
                  grad = tf.zeros_like(input_tensor)
                else:
                  grad = (input_tensor[1:] - input_tensor[:-1]) / hi
                  grad = tf.concat([grad[:1], (grad[:-1] + grad[1:]) / 2, grad[-1:]], axis=0)
            elif len(input_tensor.shape) == 2:
                if i == 0:
                  grad = (input_tensor[1:, :] - input_tensor[:-1, :]) / hi
                  grad = tf.concat([grad[:1, :], (grad[:-1, :] + grad[1:, :]) / 2, grad[-1:, :]], axis=0)
                else:
                  grad = (input_tensor[:, 1:] - input_tensor[:, :-1]) / hi
                  grad = tf.concat([grad[:, :1], (grad[:, :-1] + grad[:, 1:]) / 2, grad[:, -1:]], axis=1)
            elif len(input_tensor.shape) == 3:
              if i == 0:
                  grad = (input_tensor[1:, :, :] - input_tensor[:-1, :, :]) / hi
                  grad = tf.concat([grad[:1, :, :], (grad[:-1, :, :] + grad[1:, :, :]) / 2, grad[-1:, :, :]], axis=0)
              elif i == 1:
                  grad = (input_tensor[:, 1:, :] - input_tensor[:, :-1, :]) / hi
                  grad = tf.concat([grad[:, :1, :], (grad[:, :-1, :] + grad[:, 1:, :]) / 2, grad[:, -1:, :]], axis=1)
              else:
                  grad = (input_tensor[:, :, 1:] - input_tensor[:, :, :-1]) / hi
                  grad = tf.concat([grad[:, :, :1], (grad[:, :, :-1] + grad[:, :, 1:]) / 2, grad[:, :, -1:]], axis=2)
            else:
              raise NotImplementedError
            grads.append(grad)

        if len(grads) == 1:
            result = grads[0].numpy()
        else:
            result = tuple(g.numpy() for g in grads)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2, 4, 8], [10, 20, 40, 80]], dtype=np.float32),
        "spacing": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if isinstance(torch_result["result"], tuple):
        for torch_res, tf_res in zip(torch_result["result"], tf_result["result"]):
            assert np.allclose(torch_res, tf_res, atol=A_TOL), "Results do not match"
    else:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()