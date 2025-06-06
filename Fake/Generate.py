import numpy as np
import matplotlib.pyplot as plt

def generate_long_tail(size=1000):
    return np.random.pareto(a=2, size=size)

def long_tail_pdf(x, a=2):
    return (a * x ** (-a - 1)) * (x >= 1)

def generate_gaussian(size=1000):
    return np.abs(np.random.normal(loc=0, scale=25, size=size))
def gaussian_pdf(x, mu=0, sigma=3):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_random_noise(size=1000, low=-1, high=1):
    return np.random.uniform(low, high, size)

weights = [0.7, 0.25, 0.05]

xs = np.linspace(1, 20, 1000)
ys_long_tail = long_tail_pdf(xs)
ys_gaussian = gaussian_pdf(xs, mu = 8)
ys_random_noise = np.ones_like(xs) * (1 / (2 * (1 - (-3))))  # Uniform distribution PDF
mixture = weights[0] * ys_long_tail + weights[1] * ys_gaussian + weights[2] * ys_random_noise
plt.figure(figsize=(10, 6))
plt.plot(xs, mixture, label='Mixture Distribution', color='g')


plt.xlim(1, 20)
plt.ylim(0, 1.5)

plt.savefig('mixture_distribution.png')