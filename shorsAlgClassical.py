"""
Shor's Algorithm Implementation
================================

Shor's algorithm is a quantum algorithm for factoring large integers.
It can factor an N-bit integer in polynomial time O((log N)³).

This implementation simulates the classical parts of Shor's algorithm.
The quantum part (order finding 3c) is calculated classically.
"""

import math
from math import gcd
from typing import Tuple, List, Optional
import numpy as np

# ============================================================================
# STEP 1: CLASSICAL HELPER FUNCTIONS
# ============================================================================


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm

    Finds integers x, y such that: ax + by = gcd(a, b)

    Args:
        a: First integer
        b: Second integer

    Returns:
        Tuple of (gcd, x, y) where ax + by = gcd
    """
    if a == 0:
        return b, 0, 1
    gcd_val, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd_val, x, y


def modular_inverse(a: int, m: int) -> Optional[int]:
    """
    Find modular inverse of a modulo m

    Returns x such that (a * x) % m == 1
    Returns None if no inverse exists

    Args:
        a: The number to find inverse of
        m: The modulus

    Returns:
        The modular inverse or None if it doesn't exist
    """
    gcd_val, x, _ = extended_gcd(a % m, m)
    if gcd_val != 1:
        return None  # Modular inverse doesn't exist
    return (x % m + m) % m


def modular_exponentiation(base: int, exp: int, mod: int) -> int:
    """
    Efficiently compute (base^exp) % mod using binary exponentiation

    This is crucial for Shor's algorithm as it handles large numbers
    without overflow.

    Args:
        base: The base
        exp: The exponent
        mod: The modulus

    Returns:
        (base^exp) % mod
    """
    result = 1
    base = base % mod

    while exp > 0:
        # If exp is odd, multiply base with result
        if exp % 2 == 1:
            result = (result * base) % mod

        # Square the base and halve the exponent
        exp = exp >> 1
        base = (base * base) % mod

    return result


def is_prime(n: int) -> bool:
    """
    Miller-Rabin primality test (deterministic for n < 2^64)

    This is a probabilistic primality test that can determine if a number
    is prime with high probability. For n < 2^64, we use deterministic
    witness sets that guarantee correctness.

    The test works by:
    1. Writing n-1 = 2^s * d where d is odd
    2. Testing witnesses a using: a^d mod n
    3. Checking if result is 1 or n-1, or if (a^d)^(2^r) ≡ -1 mod n

    Witness sets are chosen to guarantee correctness for n < 2^64:
    - n < 2,047: [2]
    - n < 1,373,653: [2, 3]
    - n < 25,326,001: [2, 3, 5]
    - n < 3,825,123,056,546,413,051: [2, 3, 5, 7, 11, 13, 17, 19, 23]

    Args:
        n: Number to test for primality

    Returns:
        True if n is prime, False if n is composite
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 = 2^s * d where d is odd
    s = 0
    d = n - 1
    while d % 2 == 0:
        d //= 2
        s += 1

    # Deterministic witnesses for n < 2^64
    # These witness sets guarantee correctness for n < 2^64
    if n < 2047:
        witnesses = [2]
    elif n < 1373653:
        witnesses = [2, 3]
    elif n < 25326001:
        witnesses = [2, 3, 5]
    elif n < 3215031751:
        witnesses = [2, 3, 5, 8]
    elif n < 2152302898747:
        witnesses = [2, 3, 5, 7, 11]
    elif n < 3474749660383:
        witnesses = [2, 3, 5, 7, 11, 13]
    elif n < 341550071728321:
        witnesses = [2, 3, 5, 7, 11, 13, 17]
    else:  # n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Run Miller-Rabin test with witnesses
    for a in witnesses:
        if a >= n:
            break

        # Compute a^d mod n
        x = modular_exponentiation(a, d, n)

        # If x == 1 or x == n-1, this witness passes
        if x == 1 or x == n - 1:
            continue

        # Check if x^(2^r) ≡ -1 mod n for r = 1 to s-1
        for r in range(1, s):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            # If we never got x == n-1, n is composite
            return False

    # All witnesses passed, n is probably prime
    return True


# ============================================================================
# STEP 2: ORDER-FINDING (PERIOD FINDING) - QUANTUM PART IN PRACTICE
# ============================================================================

def find_order_classical(a: int, n: int, max_order: Optional[int] = None) -> Optional[int]:
    """
    STEP 2: Find the order r of a modulo n

    The order r is the smallest positive integer such that:
        a^r ≡ 1 (mod n)

    In a real quantum computer, this would use Quantum Phase Estimation.
    Here we use classical iteration for demonstration.

    Args:
        a: The base (must be coprime with n)
        n: The modulus
        max_order: Maximum order to check (for efficiency)

    Returns:
        The order r, or None if not found within max_order
    """
    if gcd(a, n) != 1:
        return None  # a and n must be coprime

    if max_order is None:
        max_order = n

    result = 1
    for r in range(1, max_order + 1):
        result = (result * a) % n
        if result == 1:
            return r

    return None


def find_order_optimized(a: int, n: int) -> Optional[int]:
    """
    Optimized order finding using modular exponentiation

    More efficient than naive multiplication

    Args:
        a: The base
        n: The modulus

    Returns:
        The order r

        Simply trys all r from 1 to a reasonable upper bound, checking if a^r ≡ 1 (mod n)
    """
    if gcd(a, n) != 1:
        return None

    # Start with reasonable upper bound
    max_order = min(n - 1, 1000000)

    for r in range(1, max_order + 1):
        if modular_exponentiation(a, r, n) == 1:
            return r

    return None


# ============================================================================
# STEP 3: SHOR'S ALGORITHM MAIN FUNCTION
# ============================================================================

def shors_algorithm(n: int, max_iterations: int = 10) -> Optional[Tuple[int, int]]:
    """
    SHOR'S ALGORITHM - Main Implementation

    Factors a composite integer N using the quantum algorithm steps.

    ALGORITHM STEPS:
    ================
    1. Check if N is even → factor out 2s
    2. Check if N is a perfect power → extract factors
    3. Loop max_iterations times:
        a) Pick random a where 1 < a < N
        b) Compute gcd(a, N)
           - If gcd(a, N) > 1, we found a factor! Return it.
           - If gcd(a, N) == 1, continue to step c
        c) Find the order r: smallest r where a^r ≡ 1 (mod N)
           - This is the quantum step (simulated classically here)
        d) If r is odd, go to step 3a (try new random a)
        e) If a^(r/2) ≡ -1 (mod N), go to step 3a (try new random a)
        f) Compute factors:
           x = a^(r/2)
           factor1 = gcd(x + 1, N)
           factor2 = gcd(x - 1, N)
           If either is non-trivial, return it!

    4. If no factors found after iterations, return None

    Args:
        n: The number to factor
        max_iterations: Maximum attempts to find factors

    Returns:
        A tuple (factor1, factor2) or None if factoring fails
    """

    print(f"\n{'='*70}")
    print(f"SHOR'S ALGORITHM: Factoring {n}")
    print(f"{'='*70}\n")

    # STEP 1: Handle trivial cases
    print(f"STEP 1: Check for trivial cases")
    print(f"  Is {n} even? ", end="")
    if n % 2 == 0:
        print(f"Yes → factor is 2")
        return (2, n // 2)
    print(f"No\n")

    # Check if n is prime
    if is_prime(n):
        print(f"  {n} is prime! Cannot factor.\n")
        return None

    # STEP 2: Check if n is a perfect power (n = a^b for some a, b > 1)
    print(f"STEP 2: Check if {n} is a perfect power")
    for b in range(2, int(math.log2(n)) + 1):
        a = int(round(n ** (1/b)))
        if a ** b == n:
            print(f"  {n} = {a}^{b}")
            return (a, n // a)
    print(f"  {n} is not a perfect power\n")

    # STEP 3: Main loop - quantum order finding
    print(f"STEP 3: Main loop - order finding")
    print(f"  Maximum iterations: {max_iterations}\n")

    for iteration in range(max_iterations):
        print(f"  Iteration {iteration + 1}:")

        # STEP 3a: Pick random a where 1 < a < n
        import random
        a = random.randint(2, n - 1)
        print(f"    3a) Pick random a = {a}")

        # STEP 3b: Compute gcd(a, n)
        g = gcd(a, n)
        print(f"    3b) gcd({a}, {n}) = {g}")

        if g > 1:
            print(f"    ✓ Found factor: {g}")
            print(f"\n  RESULT: {n} = {g} × {n // g}\n")
            return (g, n // g)

        # STEP 3c: Find order r such that a^r ≡ 1 (mod n)
        print(f"    3c) Finding order r where {a}^r ≡ 1 (mod {n})")
        r = find_order_optimized(a, n)

        if r is None:
            print(f"        Order not found within reasonable bound")
            continue

        print(f"        Found r = {a}^r mod {n} = 1: r = {r}")

        # STEP 3d: Check if r is even
        if r % 2 != 0:
            print(f"    3d) r is odd → try next iteration\n")
            continue

        print(f"    3d) r = {r} is even ✓")

        # STEP 3e: Check if a^(r/2) ≢ -1 (mod n)
        x = modular_exponentiation(a, r // 2, n)
        print(f"    3e) {a}^({r}//2) mod {n} = {x}")

        if x == (n - 1):  # -1 mod n
            print(f"        {x} ≡ -1 (mod {n}) → try next iteration\n")
            continue

        print(f"        {x} ≢ -1 (mod {n}) ✓")

        # STEP 3f: Compute potential factors
        print(f"    3f) Computing factors:")
        factor1 = gcd(x + 1, n)
        factor2 = gcd(x - 1, n)

        print(
            f"        gcd({a}^({r}//2) + 1, {n}) = gcd({x + 1}, {n}) = {factor1}")
        print(
            f"        gcd({a}^({r}//2) - 1, {n}) = gcd({x - 1}, {n}) = {factor2}")

        # Check if we found non-trivial factors
        if 1 < factor1 < n:
            print(f"    ✓ Found factor: {factor1}")
            print(f"\n  RESULT: {n} = {factor1} × {n // factor1}\n")
            return (factor1, n // factor1)

        if 1 < factor2 < n:
            print(f"    ✓ Found factor: {factor2}")
            print(f"\n  RESULT: {n} = {factor2} × {n // factor2}\n")
            return (factor2, n // factor2)

        print(f"    Both factors are trivial → try next iteration\n")

    print(f"\n  ✗ Failed to factor within {max_iterations} iterations\n")
    return None


# ============================================================================
# STEP 4: VERIFICATION AND EXAMPLES
# ============================================================================

def verify_factorization(n: int, factors: Tuple[int, int]) -> bool:
    """
    Verify that the factors are correct

    Args:
        n: Original number
        factors: Tuple of (factor1, factor2)

    Returns:
        True if factors[0] * factors[1] == n, False otherwise
    """
    return factors[0] * factors[1] == n


def main():
    """
    Interactive mode - allows user to input numbers to factor
    """
    print("\n" + "="*70)
    print("SHOR'S ALGORITHM - INTERACTIVE FACTORIZATION")
    print("="*70)
    print("\nThis program factors composite numbers using Shor's Algorithm.")
    print("Enter numbers to factor (or 'quit' to exit)\n")

    while True:
        try:
            # Get user input
            user_input = input(
                "Enter a number to factor (or 'quit' to exit): ").strip()

            # Check for exit condition
            if user_input.lower() == 'quit':
                print("\nThank you for using Shor's Algorithm Factorizer!")
                break

            # Try to parse the input as an integer
            n = int(user_input)

            # Validate the input
            if n < 2:
                print("Error: Please enter a number greater than 1\n")
                continue

            if n == 2:
                print("2 is prime - cannot factor\n")
                continue

            # Ask for max iterations
            try:
                max_iters = input(
                    "Enter maximum iterations (default 20): ").strip()
                if max_iters == "":
                    max_iters = 20
                else:
                    max_iters = int(max_iters)
                    if max_iters < 1:
                        print("Error: Iterations must be at least 1")
                        continue
            except ValueError:
                print("Error: Invalid number of iterations. Using default 20\n")
                max_iters = 20

            # Run Shor's algorithm
            result = shors_algorithm(n, max_iterations=max_iters)

            if result:
                print(
                    f"Verification: {result[0]} × {result[1]} = {result[0] * result[1]}")
            else:
                print("Failed to factor the number within the iteration limit.")
                print("Try increasing the maximum iterations.\n")

        except ValueError:
            print("Error: Please enter a valid integer\n")
        except KeyboardInterrupt:
            print("\n\nProgram terminated by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}\n")


if __name__ == "__main__":
    main()
