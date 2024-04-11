import numpy as np
from scipy.signal import gaussian
from skimage import restoration
import matplotlib.pyplot as plt

# Function to generate a Gaussian 1D signal
def generate_gaussian_1d(fwhm, size, sample_spacing):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return gaussian(size, std=sigma / sample_spacing)

# Generate Gaussian 1D signal
size = 1000
sample_spacing = 2.1
fwhm_signal = 50  # FWHM of the Gaussian signal in mm
gaussian_signal = generate_gaussian_1d(fwhm_signal, size, sample_spacing)

# Generate deblur kernel with full width 15mm

fwhm_kernel = 15
deblur_kernel = generate_gaussian_1d(fwhm_kernel, size, sample_spacing)
normalized_kernel = deblur_kernel / np.sum(deblur_kernel)  # Normalize the kernel

# Apply Richardson-Lucy deconvolution
num_iter = 30
# Correcting the parameter for the number of iterations
rl_deblurred_signal = restoration.richardson_lucy(gaussian_signal.reshape(-1, 1), normalized_kernel.reshape(-1, 1), num_iter=num_iter, clip=False)

# Plot the original and RL deblurred signals
plt.figure(figsize=(10, 6))
plt.plot(gaussian_signal, label='Original Gaussian Signal', alpha=0.5)
plt.plot(rl_deblurred_signal, label='RL Deblurred Signal', linestyle='-.')
plt.legend()
plt.title('Original vs RL Deblurred Gaussian 1D Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()