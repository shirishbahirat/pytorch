import numpy as np

def create_fourier_weights(sig_len):  
    "Create weights, as described above."
    kv, nv = np.mgrid[0:sig_len, 0:sig_len]
    theta = 2 * np.pi * kv * nv / sig_len
    return np.hstack([np.cos(theta), -np.sin(theta)])

sig_len = 64
x = np.random.random(size=[1, sig_len]) - 0.5

w_fourier = create_fourier_weights(sig_len)
y = np.matmul(x, w_fourier)

fft = np.fft.fft(x)
y_fft = np.hstack([fft.real, fft.imag])

print('rmse: ', np.sqrt(np.mean((y - y_fft)**2)))

import matplotlib.pyplot as plt

y_real = y[:, :sig_len]
y_imag = y[:, sig_len:]
tvals = np.arange(sig_len).reshape([-1, 1])
freqs = np.arange(sig_len).reshape([1, -1])
arg_vals = 2 * np.pi * tvals * freqs / sig_len
sinusoids = (y_real * np.cos(arg_vals) - y_imag * np.sin(arg_vals)) / sig_len
reconstructed_signal = np.sum(sinusoids, axis=1)

print('rmse:', np.sqrt(np.mean((x - reconstructed_signal)**2)))
plt.subplot(2, 1, 1)
plt.plot(x[0,:])
plt.title('Original signal')
plt.subplot(2, 1, 2)
plt.plot(reconstructed_signal)
plt.title('Signal reconstructed from sinusoids after DFT')
plt.tight_layout()
plt.show()