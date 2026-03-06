import numpy as np
import matplotlib.pyplot as plt


def get_input(prompt, default, cast=float):
    user_input = input(f"{prompt} [default: {default}]: ").strip()
    return cast(user_input) if user_input else default


def simulate_stock_prices(S0, mu, sigma, T, dt, n_sims):
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)

    S = np.zeros((n_steps, n_sims))
    S[0] = S0

    for i in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_sims)
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    return t, S


if __name__ == "__main__":
    print("=== Stock Price Simulator (Geometric Brownian Motion) ===\n")

    S0     = get_input("Initial stock price ($)",        153.77)
    mu     = get_input("Expected annual return (drift)", 0.1381)
    sigma  = get_input("Annual volatility",              0.1385)
    T      = get_input("Time horizon (years)",           1.0)
    dt     = get_input("Time step (dt)",                 0.001)
    n_sims = get_input("Number of simulations",          1000, cast=int)

    t, prices = simulate_stock_prices(S0, mu, sigma, T, dt, n_sims)

    mean_path = np.mean(prices, axis=1)
    final_prices = prices[-1]

    print(f"\n--- Results ---")
    print(f"Mean final price:   ${np.mean(final_prices):.2f}")
    print(f"Median final price: ${np.median(final_prices):.2f}")
    print(f"5th  percentile:    ${np.percentile(final_prices, 5):.2f}")
    print(f"95th percentile:    ${np.percentile(final_prices, 95):.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(t, prices, alpha=0.2, linewidth=0.5, color='steelblue')
    plt.plot(t, mean_path, 'r-', linewidth=2, label=f'Mean path')
    plt.axhline(S0, color='gray', linestyle='--', linewidth=1, label=f'S0 = ${S0:.2f}')
    plt.title(f'Stock Price Simulations (n={n_sims}, μ={mu}, σ={sigma})')
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()