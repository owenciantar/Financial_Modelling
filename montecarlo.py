import numpy as np
import matplotlib.pyplot as plt


def get_input(prompt, default, cast=float):
    """
    Prompts the user for input with a displayed default value.
    If the user presses Enter without typing, the default is used.
    `cast` converts the input string to the correct type (float by default, or int).
    """
    user_input = input(f"{prompt} [default: {default}]: ").strip()
    return cast(user_input) if user_input else default


def simulate_stock_prices(S0, mu, sigma, T, dt, n_sims):
    """
    Simulates stock price paths using Geometric Brownian Motion (GBM).

    GBM models stock prices as a continuous random process where:
      - `mu` (drift) represents the expected return over time
      - `sigma` (volatility) represents the randomness/risk in price moves

    Parameters:
        S0     - Initial stock price
        mu     - Expected annual return (drift)
        sigma  - Annual volatility (standard deviation of returns)
        T      - Total time horizon in years
        dt     - Size of each time step (smaller = more precise)
        n_sims - Number of independent simulation paths to generate

    Returns:
        t - Array of time points from 0 to T
        S - 2D array of shape (n_steps, n_sims) containing all price paths
    """

    # Calculate the total number of time steps over the horizon
    n_steps = int(T / dt)

    # Create an evenly spaced time array from 0 to T (used for plotting the x-axis)
    t = np.linspace(0, T, n_steps)

    # Initialise a 2D array to hold all simulation paths (rows = time steps, columns = simulations)
    S = np.zeros((n_steps, n_sims))

    # Set the starting price for every simulation path
    S[0] = S0

    # Iterate through each time step and compute the next price for all simulations at once
    for i in range(1, n_steps):
        # Sample a random Wiener increment dW ~ N(0, sqrt(dt)) for each simulation.
        # This represents the random "shock" to the stock price at each step.
        dW = np.random.normal(0, np.sqrt(dt), n_sims)

        # Apply the GBM formula:
        #   S[i] = S[i-1] * exp((mu - 0.5*sigma^2)*dt + sigma*dW)
        #
        # The term (mu - 0.5*sigma^2) is the Ito-corrected drift — the 0.5*sigma^2
        # adjustment accounts for the fact that returns are log-normally distributed,
        # ensuring the expected growth rate remains `mu` rather than being inflated.
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    return t, S


if __name__ == "__main__":
    print("=== Stock Price Simulator (Geometric Brownian Motion) ===\n")

    # --- Gather simulation parameters from the user (or use defaults) ---
    S0     = get_input("Initial stock price ($)",        153.77)
    mu     = get_input("Expected annual return (drift)", 0.1381)
    sigma  = get_input("Annual volatility",              0.1385)
    T      = get_input("Time horizon (years)",           1.0)
    dt     = get_input("Time step (dt)",                 0.001)
    n_sims = get_input("Number of simulations",          1000, cast=int)

    # --- Run the Monte Carlo simulation ---
    t, prices = simulate_stock_prices(S0, mu, sigma, T, dt, n_sims)

    # Compute the average price across all simulations at each time step
    mean_path = np.mean(prices, axis=1)

    # Extract the distribution of final prices (last row = end of time horizon)
    final_prices = prices[-1]

    # --- Print summary statistics for the final price distribution ---
    print(f"\n--- Results ---")
    print(f"Mean final price:   ${np.mean(final_prices):.2f}")
    print(f"Median final price: ${np.median(final_prices):.2f}")
    # 5th/95th percentiles give a 90% confidence interval for the final price
    print(f"5th  percentile:    ${np.percentile(final_prices, 5):.2f}")
    print(f"95th percentile:    ${np.percentile(final_prices, 95):.2f}")

    # --- Plot all simulation paths and the mean path ---
    plt.figure(figsize=(10, 6))

    # Draw every individual simulation path with low opacity so overlaps are visible
    plt.plot(t, prices, alpha=0.2, linewidth=0.5, color='steelblue')

    # Overlay the mean path in red so the expected trajectory stands out
    plt.plot(t, mean_path, 'r-', linewidth=2, label=f'Mean path')

    # Draw a dashed horizontal line at the starting price for reference
    plt.axhline(S0, color='gray', linestyle='--', linewidth=1, label=f'S0 = ${S0:.2f}')

    plt.title(f'Stock Price Simulations (n={n_sims}, μ={mu}, σ={sigma})')
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
