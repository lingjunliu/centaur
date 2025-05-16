import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    n_fft = input_dict["n_fft"]
    hop_length = input_dict.get("hop_length", n_fft // 4)
    win_length = input_dict.get("win_length", n_fft)
    window = input_dict.get("window", torch.ones(win_length))
    center = input_dict.get("center", True)
    normalized = input_dict.get("normalized", False)
    onesided = input_dict.get("onesided", True if n_fft != input_tensor.shape[-2] else False)
    length = input_dict.get("length", None)
    return_complex = input_dict.get("return_complex", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(window, torch.Tensor):
            window = window.cuda()

    result = torch.istft(input_tensor, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, normalized=normalized, onesided=onesided, length=length, return_complex=return_complex)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    input_tensor = tf.constant(input_dict["input"])
    n_fft = input_dict["n_fft"]
    hop_length = input_dict.get("hop_length", n_fft // 4)
    win_length = input_dict.get("win_length", n_fft)
    window = input_dict.get("window", np.ones(win_length))
    center = input_dict.get("center", True)
    normalized = input_dict.get("normalized", False)
    onesided = input_dict.get("onesided", True if n_fft != input_tensor.shape[-2] else False)
    length = input_dict.get("length", None)
    return_complex = input_dict.get("return_complex", False)

    if isinstance(window, np.ndarray):
        window = tf.constant(window, dtype=tf.float32)

    def inverse_stft(stft_matrix, frame_length, frame_step, window):
        return tf.signal.inverse_stft(
            stft_matrix,
            frame_length=frame_length,
            frame_step=frame_step,
            window_fn=tf.signal.inverse_stft_window_fn(frame_step, forward_window_fn=lambda: window),
        )

    window = tf.cast(window, dtype=tf.float32)
    input_tensor = tf.cast(input_tensor, dtype=tf.complex64)
    result = inverse_stft(input_tensor, win_length, hop_length, window)
    result = tf.cast(result, dtype=tf.float32)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 65, 10).astype(np.complex64),
        "n_fft": 128,
        "hop_length": 32,
        "win_length": 128,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()