import numpy as np
import matplotlib.pyplot as plt

# Time vector
t = np.linspace(0, 500, 25000)

# Functions to vary A, omega, phi, and B over time
def A_true(t): return 2 + 0.5 * np.sin(0.5 * t)  # Amplitude variation
def omega_true(t): return (2 * np.pi / 5) + 0.2 * np.cos(0.2 * 0)  # Frequency variation
def phi_true(t): return (np.pi / 4) + 0.1 * np.sin(0.1 * t)  # Phase variation
def B_true(t): return 1 + 0.002  * t  # Baseline offset variation

# Generating synthetic data with time-varying parameters
y_noise_free = [A_true(ti) * np.sin(omega_true(ti) * ti + phi_true(ti)) + B_true(ti) for ti in t]


# RLS Algorithm for Estimating A, omega, phi, and B
class RecursiveLeastSquaresWithOffset:
    def __init__(self, forgetting_factor=0.98, num_params=4):
        self.forgetting_factor = forgetting_factor
        self.num_params = num_params
        # Initial guesses for A, omega, phi, and B
        self.theta = np.array([np.ptp(y_noise_free) / 2, 2 * np.pi / 10, 0, np.mean(y_noise_free)])
        self.P = np.eye(num_params) * 100

    def update(self, t, y):
        A, omega, phi, B = self.theta
        phi_t = np.array([np.sin(omega * t + phi), 
                          A * t * np.cos(omega * t + phi), 
                          A * np.cos(omega * t + phi),
                          1])  # Adding constant term for B
        K = self.P @ phi_t / (self.forgetting_factor + phi_t.T @ self.P @ phi_t)
        e = y - (A * np.sin(omega * t + phi) + B)  # Prediction error including B
        self.theta += K * e
        self.P = (self.P - K[:, None] @ phi_t[None, :] @ self.P) / self.forgetting_factor
        return self.theta

# Initialize and Apply RLS for Full Parameter Estimation including B
rls_with_offset = RecursiveLeastSquaresWithOffset()
estimated_params_with_offset = []
for time, data_point in zip(t, y_noise_free):
    params = rls_with_offset.update(time, data_point)
    estimated_params_with_offset.append(params)

# Reconstructing and Plotting the Estimated Data
estimated_params_with_offset_array = np.array(estimated_params_with_offset)
estimated_y_with_offset = [estimated_params_with_offset_array[i, 0] * np.sin(estimated_params_with_offset_array[i, 1] * t[i] + estimated_params_with_offset_array[i, 2]) + estimated_params_with_offset_array[i, 3]
                           for i in range(len(t))]
plt.figure(figsize=(12, 6))
plt.plot(t, y_noise_free, label='Noise-Free Real Data', color='blue')
plt.plot(t, estimated_y_with_offset, label='Estimated Data with Offset', color='orange', linestyle='--')
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("Noise-Free Real Data vs Estimated Data with Offset")
plt.legend()
plt.show()
