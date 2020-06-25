# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set()


# %%
f0 = 0.3817

xp = np.arange(32)
yp = np.sin((2 * np.pi * f0) * xp)

plt.scatter(xp, yp)


# %%
x = np.arange(31, step=0.01)
y = np.zeros(np.shape(x))


# %%
for ii in np.arange(np.size(x)):
    idx = int(round(x[ii]))
    y[ii] = yp[idx]

plt.plot(x, y)
plt.scatter(xp, yp)


# %%
for ii in np.arange(np.size(x)):
    dec, idx = math.modf(x[ii])
    idx = int(idx)
    y[ii] = yp[idx] + (yp[idx + 1] - yp[idx]) * dec

plt.plot(x, y)
plt.scatter(xp, yp)


# %%
size_parameter = 5

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

plt.plot(x, y)
plt.scatter(xp, yp)


# %%
plt.plot(x, window_size)

# %%
size_parameter = 5

for ii in np.arange(np.size(x)):
    lower_idx = int(np.ceil(x[ii] - size_parameter))
    upper_idx = int(np.ceil(x[ii] + size_parameter))
    if lower_idx < 0:
        lower_idx = 0
    if upper_idx > np.size(xp):
        upper_idx = int(np.size(xp))
    sinc_args = x[ii] - np.arange(start=lower_idx, stop=upper_idx)
    y[ii] = np.sum(yp[lower_idx:upper_idx] * np.sinc(sinc_args) *
                   (0.5 + 0.5 * np.cos(np.pi * sinc_args / size_parameter)))

plt.plot(x, y)
plt.scatter(xp, yp)


# %%
size_parameter = 5

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

plt.plot(x, y)
plt.scatter(xp, yp)


# %%
