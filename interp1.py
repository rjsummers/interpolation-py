# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import libraries.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set()


# %%
# Generate a small dataset to interpolate.
f0 = 0.3817
Np = 33

xp = np.arange(Np)
yp = np.sin((2 * np.pi * f0) * xp)

plt.scatter(xp, yp)
plt.title("Original data")


# %%
# Create interpolation output variables.
x = np.arange(Np - 1, step=0.01)


# %%
# Nearest-neighbor interpolation.
def interp1_nearest_neighbor(xp, yp, x):
    y = np.zeros(np.shape(x))
    for ii in np.arange(np.size(x)):
        idx = int(round(x[ii]))
        y[ii] = yp[idx]
    return y


y = interp1_nearest_neighbor(xp, yp, x)

plt.plot(x, y)
plt.scatter(xp, yp)
plt.title("Nearest neighbor")


# %%
# Linear interpolation.
def interp1_linear(xp, yp, x):
    y = np.zeros(np.shape(x))
    for ii in np.arange(np.size(x)):
        dec, idx = math.modf(x[ii])
        idx = int(idx)
        y[ii] = yp[idx] + (yp[idx + 1] - yp[idx]) * dec
    return y


y = interp1_linear(xp, yp, x)

plt.plot(x, y)
plt.scatter(xp, yp)
plt.title("Linear interpolation")


# %%
# Rectangular windowed sinc interpolation.
def interp1_wsinc_rect(xp, yp, x, size_parameter):
    window_size = np.zeros(np.size(y))
    for ii in np.arange(np.size(x)):
        lower_idx = int(np.ceil(x[ii] - size_parameter))
        upper_idx = int(np.ceil(x[ii] + size_parameter))
        if lower_idx < 0:
            lower_idx = 0
        if upper_idx > np.size(xp):
            upper_idx = int(np.size(xp))
        window_size[ii] = upper_idx - lower_idx
        sinc_args = x[ii] - np.arange(start=lower_idx, stop=upper_idx)
        y[ii] = np.sum(yp[lower_idx:upper_idx] * np.sinc(sinc_args))
    return y


size_parameter = 5
y = interp1_wsinc_rect(xp, yp, x, size_parameter)

plt.plot(x, y)
plt.scatter(xp, yp)
plt.title("Rect. windowed sinc")


# %%
# Hann-windowed sinc interpolation.
def interp1_wsinc_hann(xp, yp, x, size_parameter):
    for ii in np.arange(np.size(x)):
        lower_idx = int(np.ceil(x[ii] - size_parameter))
        upper_idx = int(np.ceil(x[ii] + size_parameter))
        if lower_idx < 0:
            lower_idx = 0
        if upper_idx > np.size(xp):
            upper_idx = int(np.size(xp))
        sinc_args = x[ii] - np.arange(start=lower_idx, stop=upper_idx)
        y[ii] = np.sum(yp[lower_idx:upper_idx] * np.sinc(sinc_args) *
                       (0.5 + 0.5 * np.cos(np.pi * sinc_args /
                                           size_parameter)))
    return y


y = interp1_wsinc_hann(xp, yp, x, size_parameter)

plt.plot(x, y)
plt.scatter(xp, yp)
plt.title("Hann windowed sinc")


# %%
# Lanczos-windowed sinc interpolation.
def interp1_wsinc_lanczos(xp, yp, x, size_parameter):
    y = np.zeros(np.shape(x))
    for ii in np.arange(np.size(x)):
        lower_idx = int(np.ceil(x[ii] - size_parameter))
        upper_idx = int(np.ceil(x[ii] + size_parameter))
        if lower_idx < 0:
            lower_idx = 0
        if upper_idx > np.size(xp):
            upper_idx = int(np.size(xp))
        sinc_args = x[ii] - np.arange(start=lower_idx, stop=upper_idx)
        y[ii] = np.sum(yp[lower_idx:upper_idx] * np.sinc(sinc_args) *
                       np.sinc(sinc_args / size_parameter))
    return y


y = interp1_wsinc_lanczos(xp, yp, x, size_parameter)

plt.plot(x, y)
plt.scatter(xp, yp)
plt.title("Lanczos windowed sinc")


# %%
# FFT-based interpolation.
# TODO: Add a linear interpolation step to allow arbitrary query points
Norig = np.shape(xp)[0]
Nint = 1024
pad_size = Nint - Norig

ys = np.fft.fft(yp)
ys = np.insert(ys, int(np.ceil(Norig/2)), np.zeros(pad_size))
yr = np.fft.ifft(ys) * (Nint / Norig)

xr = np.arange(0, Nint) * Norig / Nint

plt.plot(xr, yr)
plt.scatter(xp, yp)
plt.title("FFT-based interpolation")


# %%
# Generate a larger dataset for error calculations.
f0 = 0.3817
Np = 21232

xp = np.arange(Np)
yp = np.sin((2 * np.pi * f0) * xp)

Nint = 4096 * 2160
x = np.random.rand(Nint) * (21232 - 1)

# %%

y = interp1_wsinc_lanczos(xp, yp, x, 11)
mse = ((y - np.sin((2 * np.pi * f0) * x))**2).mean()

# %%
