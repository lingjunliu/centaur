import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

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

    result = torch.istft(
        input_tensor,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        length=length,
        return_complex=return_complex,
    )

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

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        window = tf.constant(window, dtype=input_tensor.dtype.real_dtype)

        input_shape = tf.shape(input_tensor)
        num_frames = input_shape[-1]

        frames = tf.transpose(input_tensor, perm=[0, 2, 1]) if len(input_shape) > 2 else tf.transpose(input_tensor, perm=[1, 0])
        frames = tf.cast(frames, dtype=tf.complex64)

        signal = tf.signal.inverse_stft(
            frames,
            frame_length=win_length,
            frame_step=hop_length,
            fft_length=n_fft,
            window_fn=lambda frame_length, dtype: window,
        )

        if center:
            pad_total = n_fft
            signal_len = tf.shape(signal)[-1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            signal = tf.pad(signal, [[0, 0], [pad_left, pad_right]])
            if length is not None:
                signal = signal[:length]

        elif length is not None:
            signal = tf.pad(signal, [[0, max(0, length - tf.shape(signal)[0])]])
            signal = signal[:length]

        result = signal.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, (1024 // 2) + 1, 100).astype(np.complex64),
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "center": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    min_len = min(torch_result["result"].shape[-1], tf_result["result"].shape[-1])

    assert np.allclose(torch_result["result"][..., :min_len], tf_result["result"][..., :min_len], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()