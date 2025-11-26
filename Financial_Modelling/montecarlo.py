import numpy as np
import matplotlib.pyplot as plt

def simulate_stock_prices(S0=153.77, mu=0.1381, sigma=0.1385, T=1, dt=0.001, n_sims=1000):
    """
    Simulate stock prices using geometric Brownian motion.
    
    Parameters:
    S0: initial stock price
    mu: expected return (drift)
    sigma: volatility
    T: time horizon in years
    dt: time step
    n_sims: number of simulations
    """

    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    
    # Initialize array for stock prices
    S = np.zeros((n_steps, n_sims))
    S[0] = S0
    
    # Simulate prices for each path
    for i in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_sims)
        S[i] = S[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)
    
    return t, S

# Run the simulation to create t and prices
t, prices = simulate_stock_prices()

# Plot stock simulations
plt.figure(figsize=(10, 6))
plt.plot(t, prices, alpha=0.3, linewidth=0.5)
plt.plot(t, np.mean(prices, axis=1), 'r-', linewidth=2, label='Mean')
plt.title('Stock Price Simulations')
plt.xlabel('Time (years)')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()