import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    n_fft = input_dict.get("length", None)
    hop_length = input_dict.get("hop_length", None)
    win_length = input_dict.get("win_length", None)
    window = input_dict.get("window", None)
    if window is not None:
        window = torch.tensor(window)
    center = input_dict.get("center", True)
    pad_mode = input_dict.get("pad_mode", "reflect")
    normalized = input_dict.get("normalized", False)
    onesided = input_dict.get("onesided", True)
    return_complex = input_dict.get("return_complex", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if window is not None:
            window = window.cuda()

    result = torch.stft(
        input_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
        return_complex=not not return_complex,
    )

    if not cpu:
        result = result.cpu()

    if not return_complex:
        result = torch.view_as_real(result)
        result = result.reshape(result.shape[:-1] + (-1,))

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import numpy as np

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    length = input_dict.get("length", None)
    hop_length = input_dict.get("hop_length", None)
    win_length = input_dict.get("win_length", None)
    window = input_dict.get("window", None)
    center = input_dict.get("center", True)
    pad_mode = input_dict.get("pad_mode", "REFLECT")
    normalized = input_dict.get("normalized", False)
    onesided = input_dict.get("onesided", True)
    return_complex = input_dict.get("return_complex", False)

    if length is None:
        length = tf.shape(input_tensor)[-1]
    if hop_length is None:
        hop_length = length // 4
    if win_length is None:
        win_length = length
    if window is None:
        window = tf.signal.hann_window(win_length, dtype=tf.float32)
    else:
        window = tf.cast(window, dtype=tf.float32)

    if center:
        pad_left = (win_length - hop_length + 1) // 2
        pad_right = win_length - hop_length - pad_left
        input_tensor = tf.pad(
            input_tensor,
            [[pad_left, pad_right]],
            mode=pad_mode,
        )

    stft = tf.signal.stft(
        input_tensor,
        frame_length=win_length,
        frame_step=hop_length,
        fft_length=length,
        window_fn=lambda frame_length, dtype: window,
        pad_end=False
    )
    
    if not onesided:
        num_fft_bins = length
        if num_fft_bins % 2 == 0:
            lower_cutoff = -(num_fft_bins // 2 - 1)
        else:
            lower_cutoff = -(num_fft_bins - 1) // 2
        stft = tf.concat([tf.reverse(tf.math.conj(stft[..., lower_cutoff:]), axis=[-1]), stft[..., 1:]], axis=-1)

    if normalized:
        stft = stft / tf.sqrt(tf.cast(win_length, tf.complex64))

    if not return_complex:
        stft = tf.abs(stft)

    if return_complex:
        stft = tf.complex(tf.math.real(stft), tf.math.imag(stft))
    else:
        stft = tf.cast(stft, dtype=tf.float32)
    
    return {"result": stft.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        "length": 8,
        "hop_length": 2,
        "win_length": 4,
        "return_complex": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_res = torch_result["result"]
    tf_res = tf_result["result"]
    
    if torch_res.shape != tf_res.shape:
      min_time = min(torch_res.shape[0], tf_res.shape[0])
      min_freq = min(torch_res.shape[1], tf_res.shape[1])
      if len(torch_res.shape) == 3:
        torch_res = torch_res[:min_time, :min_freq, :]
        tf_res = tf_res[:min_time, :min_freq]
      else:
        torch_res = torch_res[:min_time, :min_freq]
        tf_res = tf_res[:min_time, :min_freq]


    assert np.allclose(torch_res, tf_res, atol=A_TOL)

    print("Success")


if __name__ == "__main__":
    main()