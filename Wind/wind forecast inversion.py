
"""
@Date: 2025/10/29

@Author: DREAM_DXC

@Summary: Wind Power Forecast Inversion Program

@Note: If you aim to analyze the impact of prediction accuracy on power systems, you typically need to construct a
dataset of renewable energy forecasts with varying levels of accuracy. However, such comprehensive datasets are
difficult to obtain in practice. To address this, this project proposes a dynamic weighted smoothing approach to
generate forecast data under specified prediction errors, enabling a systematic analysis of the influence of prediction
accuracy on power systems.

"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default="windpower.xlsx", help='Path to the input file')
parser.add_argument("--target_error", type=float, default=0.12, help="Target of forecast error")
parser.add_argument("--cap", type=float, default=1, help="Capacity of wind farm")
parser.add_argument("--max_iter", type=int, default=500, help="Maximum number of iterations")
parser.add_argument("--daily_step", type=int, default=24, help="Daily time step")
parser.add_argument("--max_window_size", type=int, default=50, help="Maximum number of smooth window size")
parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
parser.add_argument("--tol", type=int, default=1e-4, help="Error convergence threshold")
parser.add_argument("--seed", default=1, type=int)

args = parser.parse_args()
print(args)

def moving_average_smooth(data, window_size, mode):

    """
    Parameters:
    data: Input data sequence
    window_size: Size of the smoothing window (must be odd)
    mode: Smoothing mode
        'temporal' - Smooth across adjacent time points (used for wind power)
        'daily' - Smooth across corresponding time points of different days (used for solar power)
    Returns:
    smoothed_data: The smoothed data
    """

    n = len(data)
    smoothed_data = np.zeros(n)

    # Ensure the window size is odd
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2

    if mode == 'temporal':
        # Wind power mode: smooth across adjacent time points
        for i in range(n):
            if i < half_window:
                # Left boundary
                start_idx = 0
                end_idx = i + half_window + 1
            elif i >= n - half_window:
                # Right boundary
                start_idx = i - half_window
                end_idx = n
            else:
                # Middle section
                start_idx = i - half_window
                end_idx = i + half_window + 1

            window_data = data[start_idx:end_idx]
            smoothed_data[i] = np.mean(window_data)

    elif mode == 'daily':
        # Solar power mode: smooth across corresponding time points of different days

        hours_per_day = args.daily_step
        days_per_year = n // hours_per_day

        for i in range(n):
            current_hour = i % hours_per_day
            current_day = i // hours_per_day

            # Collect data from the same hour across neighboring days
            neighbor_days = []

            for day_offset in range(-half_window, half_window + 1):
                target_day = current_day + day_offset

                # Check if the target day is within valid range
                if 0 <= target_day < days_per_year:
                    target_idx = target_day * hours_per_day + current_hour
                    neighbor_days.append(data[target_idx])

            # Calculate the mean
            if neighbor_days:
                smoothed_data[i] = np.mean(neighbor_days)
            else:
                smoothed_data[i] = data[i]  # Use original value if no neighbors are available

    return smoothed_data

def wind_kernel_width_traverse(wind_power, max_window_size=args.max_window_size):
    """
    Kernel width traversal analysis for wind power.

    Parameters:
    wind_power: measured wind power time series
    max_window_size: Maximum window size for traversal (default from args)

    Returns:
    results: 2D array, first column is kernel width, second column is corresponding MAE
    """

    kernel_widths = list(range(3, max_window_size, 2))

    results = []

    print("Starting kernel width traversal analysis for wind power...")
    for width in kernel_widths:
        # Apply moving average smoothing across adjacent time points to wind power
        smoothed_power = moving_average_smooth(wind_power, width, mode='temporal')

        # Calculate MAE between smoothed and original series
        mae = mean_absolute_error(wind_power, smoothed_power)

        results.append([width, mae])

        if len(results) % 10 == 0:
            print(f"Completed {len(results)}/{len(kernel_widths)}: Width={width}, MAE={mae:.6f}")

    # Convert to numpy array
    results_array = np.array(results)

    # Save to Excel file if requested
    xlsfile = pd.ExcelWriter("wind_kernel_analysis.xlsx")
    df = pd.DataFrame(results_array)
    df.to_excel(xlsfile,header=['Kernel_Width', 'MAE'],index=False)
    xlsfile.close()

    return results_array

def find_optimal_kernel_width(kernel_results, target_err):
    """
    Find the optimal kernel width based on traversal results and target error.

    Parameters:
    kernel_results: 2D array [kernel_width, MAE]
    target_err: Target MAE threshold

    Returns:
    optimal_width: The smallest width where MAE < target_err, or None if not found
    """

    # Sort by kernel width ascending
    sorted_results = kernel_results[kernel_results[:, 0].argsort()]

    for width, mae in sorted_results:
        if mae >= target_err:
            return int(width), mae  # First width that meets the accuracy requirement

    return None  # No width meets the target error

def adaptive_weighted_average_wind(data, target_err, max_iter=args.max_iter, learning_rate=args.lr, tol=args.tol):
    """
    Adaptive weighted moving average smoothing algorithm for wind power.

    Parameters:
    data: Raw wind power time series
    target_err: Target MAE (Mean Absolute Error)
    max_iter: Maximum number of iterations
    learning_rate: Learning rate for gradient adjustment
    tol: Convergence tolerance

    Returns:
    smoothed_data: Smoothed data
    a_value: Final weight parameter 'a'
    final_mae: Final MAE value
    """

    n = len(data)

    kernel_results = wind_kernel_width_traverse(data)

    plot_traverse_results(kernel_results)

    # Find the smallest kernel width that meets the target error
    window_size, uniform_mae = find_optimal_kernel_width(kernel_results, target_err)

    print(f"Selected kernel width: {window_size}, "
          f"MAE with uniform weights: {uniform_mae:.6f}, "
          f"Target MAE: {target_err:.6f}")

    half_window = window_size // 2

    # Define adjustable weight function
    def weight_function(distances, a):
        weights = np.exp(-a * distances)
        return weights / np.sum(weights)

    a = 0.0  # Start from uniform weights (a=0 gives equal weights)

    # Store iteration history
    history = {'a': [], 'mae': []}

    # Gradient descent optimization
    for iteration in range(max_iter):
        smoothed_data = np.zeros(n)

        # Compute smoothed sequence with current weights
        for i in range(n):
            # Compute distances from current point to points in the window
            if i < half_window:
                start_idx = 0
                end_idx = i + half_window + 1
                distances = np.abs(np.arange(start_idx, end_idx) - i)
            elif i >= n - half_window:
                start_idx = i - half_window
                end_idx = n
                distances = np.abs(np.arange(start_idx, end_idx) - i)
            else:
                start_idx = i - half_window
                end_idx = i + half_window + 1
                distances = np.abs(np.arange(start_idx, end_idx) - i)

            distances = distances / np.max(distances)  # Normalize distances
            weights = weight_function(distances, a)
            window_data = data[start_idx:end_idx]

            smoothed_data[i] = np.sum(window_data * weights)

        # Compute current MAE
        current_mae = mean_absolute_error(data, smoothed_data)

        # Record history
        history['a'].append(a)
        history['mae'].append(current_mae)

        # Check for convergence
        if abs(current_mae - target_err) < tol:
            print(f"Converged after {iteration + 1} iterations")
            print(f"Final parameter a = {a:.6f}, MAE = {current_mae:.6f}")
            return smoothed_data

        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: a = {a:.6f}, MAE = {current_mae:.6f}")

        # Compute gradient (numerical approximation)
        delta = 0.001
        a_plus = a + delta

        # Compute MAE when a = a + delta
        mae_plus = 0
        for i in range(n):
            if i < half_window:
                start_idx = 0
                end_idx = i + half_window + 1
                distances = np.abs(np.arange(start_idx, end_idx) - i)
            elif i >= n - half_window:
                start_idx = i - half_window
                end_idx = n
                distances = np.abs(np.arange(start_idx, end_idx) - i)
            else:
                start_idx = i - half_window
                end_idx = i + half_window + 1
                distances = np.abs(np.arange(start_idx, end_idx) - i)

            distances = distances / np.max(distances)
            weights_plus = weight_function(distances, a_plus)
            window_data = data[start_idx:end_idx]

            mae_plus += abs(data[i] - np.sum(window_data * weights_plus))

        mae_plus /= n  # Convert sum of absolute errors to mean absolute error

        # Compute gradient
        gradient = (mae_plus - current_mae) / delta

        # Adjust parameter 'a' based on error
        # If current MAE > target, we need to reduce MAE -> increase 'a' for sharper weights
        # If current MAE < target, we need to increase MAE (less smoothing) -> decrease 'a'
        if current_mae > target_err:
            # Need to reduce MAE -> increase 'a'
            a += learning_rate * abs(gradient)
        else:
            # Need to increase MAE (reduce smoothing) -> decrease 'a'
            a -= learning_rate * abs(gradient)

    print(f"Reached maximum iterations {max_iter}, did not fully converge")
    print(f"Final parameter a = {a:.6f}, MAE = {current_mae:.6f}")

    return smoothed_data

def plot_traverse_results(traverse_data):

    plt.plot(traverse_data[:, 0], traverse_data[:, 1], 'b-', linewidth=2)
    plt.xlabel('windows size')
    plt.ylabel('MAE')
    plt.title('wind traverse smooth plot')

    plt.savefig('traverse_results.png', dpi=300)
    plt.show()

def forecast_inversion_plot(wind_power,smoothed_data):

    plt.plot(wind_power[:2000], 'k-', label='Original wind power ', linewidth=1.0)
    plt.plot(smoothed_data[:2000], 'r-', label='Smoothed wind power ', linewidth=1.0)
    plt.xlabel('time')
    plt.ylabel('wind power')

    plt.savefig('adaptive_smoothing_comparison.png', dpi=300)
    plt.show()

    return None


if __name__ == "__main__":

    df = pd.read_excel(args.filename)
    wind_power = df.iloc[:, 1].values/args.cap

    smoothed_data = adaptive_weighted_average_wind(wind_power, args.target_error)

    forecast_inversion_plot(wind_power,smoothed_data)