# Stock Price Simulator — Geometric Brownian Motion

A Monte Carlo stock price simulator built in Python using Geometric Brownian Motion (GBM). Simulates thousands of potential price paths for a given stock and outputs key statistics and a visualisation.

## How It Works

The simulator models stock price evolution using the GBM stochastic differential equation:

$$dS = \mu S \, dt + \sigma S \, dW$$

Which in discrete form becomes:

$$S_{t+dt} = S_t \cdot \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \sqrt{dt} \cdot Z\right)$$

where $Z \sim \mathcal{N}(0, 1)$ is a standard normal random variable.

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib
