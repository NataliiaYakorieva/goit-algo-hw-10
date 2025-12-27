import numpy as np
import scipy.integrate as spi
from typing import Callable, Union


def quadratic_function(x: Union[float, np.ndarray]
                       ) -> Union[float, np.ndarray]:
    """
    Function to integrate: f(x) = x^2.
    Args:
        x (float or np.ndarray): Input value(s).
    Returns:
        float or np.ndarray: Output of the function f(x) = x^2.
    """
    return x ** 2


def monte_carlo_integral(
        func: Callable[[np.ndarray], np.ndarray],
        lower_bound: float,
        upper_bound: float,
        num_samples: int
) -> float:
    """
    Estimates the definite integral of a function using the Monte Carlo method.
    Args:
        func (Callable): Function to integrate.
        lower_bound (float): Lower bound of integration.
        upper_bound (float): Upper bound of integration.
        num_samples (int): Number of random samples.
    Returns:
        float: Estimated value of the integral.
    """
    x_rand: np.ndarray = np.random.uniform(
        lower_bound, upper_bound, num_samples)
    y_rand: np.ndarray = func(x_rand)
    return float((upper_bound - lower_bound) * np.mean(y_rand))


def main() -> None:
    """
    Calculates the integral of quadratic_function(x) = x^2 from 0 to 2 using both Monte Carlo and quad methods.
    Prints the results and their difference.
    """
    lower_bound: float = 0
    upper_bound: float = 2
    num_samples: int = 100_000

    # Monte Carlo estimation
    integral_mc: float = monte_carlo_integral(
        quadratic_function, lower_bound, upper_bound, num_samples
    )
    print("Monte Carlo integral:", integral_mc)

    # Quad (analytical) calculation
    result, error = spi.quad(quadratic_function, lower_bound, upper_bound)
    print("Quad integral:", result, "Absolute error:", error)

    # Comparison
    print("Difference:", abs(integral_mc - result))


if __name__ == "__main__":
    main()
