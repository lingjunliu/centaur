import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    hx = torch.tensor(input_dict["hx"])
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
        w_ih = w_ih.cuda()
        w_hh = w_hh.cuda()
        b_ih = b_ih.cuda()
        b_hh = b_hh.cuda()
        scale_ih = scale_ih.cuda()
        scale_hh = scale_hh.cuda()
        zero_point_ih = zero_point_ih.cuda()
        zero_point_hh = zero_point_hh.cuda()

    def emulate_quantize(x, scale, zero_point):
        x_div_scale = x / scale
        x_add_zero_point = x_div_scale + zero_point.float()
        x_rounded = torch.round(x_add_zero_point)
        x_clamped = torch.clamp(x_rounded, 0, 255)
        x_quantized = x_clamped.byte()
        return x_quantized

    def emulate_dequantize(x, scale, zero_point):
        x_float = x.float()
        x_sub_zero_point = x_float - zero_point.float()
        x_mul_scale = x_sub_zero_point * scale
        return x_mul_scale

    q_w_ih = emulate_quantize(w_ih, scale_ih, zero_point_ih)
    dq_w_ih = emulate_dequantize(q_w_ih, scale_ih, zero_point_ih)
    q_w_hh = emulate_quantize(w_hh, scale_hh, zero_point_hh)
    dq_w_hh = emulate_dequantize(q_w_hh, scale_hh, zero_point_hh)
    
    gate_input = torch.matmul(input_tensor, dq_w_ih) + b_ih
    gate_hidden = torch.matmul(hx, dq_w_hh) + b_hh
    gates = gate_input + gate_hidden
    
    new_hx = torch.relu(gates)

    if not cpu:
        new_hx = new_hx.cpu()

    return {"result_hx": new_hx.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        hx = tf.constant(input_dict["hx"])
        w_ih = tf.constant(input_dict["w_ih"])
        w_hh = tf.constant(input_dict["w_hh"])
        b_ih = tf.constant(input_dict["b_ih"])
        b_hh = tf.constant(input_dict["b_hh"])
        scale_ih = tf.constant(input_dict["scale_ih"])
        scale_hh = tf.constant(input_dict["scale_hh"])
        zero_point_ih = tf.constant(input_dict["zero_point_ih"], dtype=tf.int32)
        zero_point_hh = tf.constant(input_dict["zero_point_hh"], dtype=tf.int32)

        def emulate_quantize(x, scale, zero_point):
            x_div_scale = tf.divide(x, scale)
            x_add_zero_point = tf.add(x_div_scale, tf.cast(zero_point, dtype=x.dtype))
            x_rounded = tf.round(x_add_zero_point)
            x_clamped = tf.clip_by_value(x_rounded, 0, 255)
            x_quantized = tf.cast(x_clamped, dtype=tf.uint8)
            return x_quantized

        def emulate_dequantize(x, scale, zero_point):
            x_float = tf.cast(x, dtype=tf.float32)
            x_sub_zero_point = tf.subtract(x_float, tf.cast(zero_point, dtype=tf.float32))
            x_mul_scale = tf.multiply(x_sub_zero_point, scale)
            return x_mul_scale

        q_w_ih = emulate_quantize(w_ih, scale_ih, zero_point_ih)
        dq_w_ih = emulate_dequantize(q_w_ih, scale_ih, zero_point_ih)
        q_w_hh = emulate_quantize(w_hh, scale_hh, zero_point_hh)
        dq_w_hh = emulate_dequantize(q_w_hh, scale_hh, zero_point_hh)
        
        gate_input = tf.matmul(input_tensor, dq_w_ih) + b_ih
        gate_hidden = tf.matmul(hx, dq_w_hh) + b_hh
        gates = gate_input + gate_hidden
        
        new_hx = tf.nn.relu(gates)

        result_hx = new_hx.numpy()

    return {"result_hx": result_hx}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056], [0.0302, 1.1985, 1.4506, -0.7056]], dtype=np.float32),
        "hx": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "w_ih": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
        "w_hh": np.array([[0.9, 1.0], [1.1, 1.2]], dtype=np.float32),
        "b_ih": np.array([0.01, 0.02], dtype=np.float32),
        "b_hh": np.array([0.03, 0.04], dtype=np.float32),
        "scale_ih": np.array([0.1], dtype=np.float32),
        "scale_hh": np.array([0.2], dtype=np.float32),
        "zero_point_ih": np.array([0], dtype=np.int32),
        "zero_point_hh": np.array([0], dtype=np.int32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result_hx"], tf_result["result_hx"], atol=A_TOL), "result_hx does not match"

    print("Success")


if __name__ == "__main__":
    main()