import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn import LSTMCell

    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])
    cx = torch.tensor(input_dict["cx"])
    w_ih = torch.tensor(input_dict["w_ih"])
    w_hh = torch.tensor(input_dict["w_hh"])
    b_ih = torch.tensor(input_dict["b_ih"])
    b_hh = torch.tensor(input_dict["b_hh"])
    scale_ih = torch.tensor(input_dict["scale_ih"])
    scale_hh = torch.tensor(input_dict["scale_hh"])
    zero_point_ih = torch.tensor(input_dict["zero_point_ih"], dtype=torch.int32)
    zero_point_hh = torch.tensor(input_dict["zero_point_hh"], dtype=torch.int32)

    if not cpu:
        input_tensor = input_tensor.cuda()
        hx = hx.cuda()
        cx = cx.cuda()
        w_ih = w_ih.cuda()
        w_hh = w_hh.cuda()
        b_ih = b_ih.cuda()
        b_hh = b_hh.cuda()
        scale_ih = scale_ih.cuda()
        scale_hh = scale_hh.cuda()
        zero_point_ih = zero_point_ih.cuda()
        zero_point_hh = zero_point_hh.cuda()

    lstm_cell = LSTMCell(input_size=input_tensor.shape[1], hidden_size=hx.shape[1])

    lstm_cell.weight_ih = torch.nn.Parameter(w_ih)
    lstm_cell.weight_hh = torch.nn.Parameter(w_hh)
    lstm_cell.bias_ih = torch.nn.Parameter(b_ih)
    lstm_cell.bias_hh = torch.nn.Parameter(b_hh)

    result = lstm_cell(input_tensor, (hx, cx))

    if not cpu:
        result = (result[0].cpu(), result[1].cpu())

    return {"result_h": result[0].detach().numpy(), "result_c": result[1].detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        hx = tf.constant(input_dict["hx"], dtype=tf.float32)
        cx = tf.constant(input_dict["cx"], dtype=tf.float32)
        w_ih = tf.constant(input_dict["w_ih"], dtype=tf.float32)
        w_hh = tf.constant(input_dict["w_hh"], dtype=tf.float32)
        b_ih = tf.constant(input_dict["b_ih"], dtype=tf.float32)
        b_hh = tf.constant(input_dict["b_hh"], dtype=tf.float32)
        scale_ih = tf.constant(input_dict["scale_ih"], dtype=tf.float32)
        scale_hh = tf.constant(input_dict["scale_hh"], dtype=tf.float32)
        zero_point_ih = tf.constant(input_dict["zero_point_ih"], dtype=tf.int32)
        zero_point_hh = tf.constant(input_dict["zero_point_hh"], dtype=tf.int32)

        def lstm_cell(input_tensor, hx, cx, w_ih, w_hh, b_ih, b_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh):

            input_tensor = tf.expand_dims(input_tensor, 0)
            hx = tf.expand_dims(hx, 0)
            cx = tf.expand_dims(cx, 0)
            
            gates = tf.matmul(input_tensor, tf.transpose(w_ih)) + tf.matmul(hx, tf.transpose(w_hh)) + b_ih + b_hh

            n_units = w_ih.shape[0] // 4
            ingate = gates[:, :, :n_units]
            forgetgate = gates[:, :, n_units:2*n_units]
            cellgate = gates[:, :, 2*n_units:3*n_units]
            outgate = gates[:, :, 3*n_units:]

            ingate = tf.sigmoid(ingate)
            forgetgate = tf.sigmoid(forgetgate)
            cellgate = tf.tanh(cellgate)
            outgate = tf.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * tf.tanh(cy)

            return hy[0, 0], cy[0, 0]

        result_h, result_c = lstm_cell(input_tensor, hx, cx, w_ih, w_hh, b_ih, b_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh)

        result_h = result_h.numpy()
        result_c = result_c.numpy()

    return {"result_h": result_h, "result_c": result_c}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2]], dtype=np.float32),
        "hx": np.array([[0.3, 0.4]], dtype=np.float32),
        "cx": np.array([[0.5, 0.6]], dtype=np.float32),
        "w_ih": np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2], [1.3, 1.4]], dtype=np.float32),
        "w_hh": np.array([[1.5, 1.6], [1.7, 1.8], [1.9, 2.0], [2.1, 2.2]], dtype=np.float32),
        "b_ih": np.array([2.3, 2.4, 2.5, 2.6], dtype=np.float32),
        "b_hh": np.array([2.7, 2.8, 2.9, 3.0], dtype=np.float32),
        "scale_ih": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "scale_hh": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "zero_point_ih": np.array([0, 0, 0, 0], dtype=np.int32),
        "zero_point_hh": np.array([0, 0, 0, 0], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result_h"], tf_result["result_h"], atol=A_TOL), "Results do not match for result_h"
    assert np.allclose(torch_result["result_c"], tf_result["result_c"], atol=A_TOL), "Results do not match for result_c"

    print("Success")

if __name__ == "__main__":
    main()