import numpy as np

def polyfute(m, n):
    """
	Compute the list of coefficients K_mn of the Zernike polynomials.
	K_mn(s) is the weighting coefficient of r^(n-2s).
    I use the following relations between the K_mn to facilitate the
	computation and avoid cumbersome factorials:
    K_mn(s+1) =  K_mn(s) * ((n+m)/2-s)*((n-m)/2-s)/(s+1)/(n-s)
    and
	K_mn(0) = n! / ((n+m)/2)! / ((n-m)/2)!
    """
	# Array of the K_mn (it's called 'a' .. yes .. I know.. but that was shorter)
    a = np.zeros(n + 1, dtype=np.float32)

    # Computation of K_mn(0)
    st = 2  # start index for dividing by ((n-m)/2)!
    coef = 1
    for i in range((n + m) // 2 + 1, n + 1):  # computation of  n! / ((n+m)/2)!
        if (st <= ((n - m) // 2) and i % st == 0):
            j = i // st
            st = st + 1
            coef = coef * j
        else:
            coef = coef * i

    # division by ((n-m)/2)! (has already been partially done)
    for i in range(st, (n - m) // 2 + 1):
        coef = coef / i

    a[n] = coef  # for K_nm(0)

    for i in range(1, (n - m) // 2 + 1):
        coef = -coef * ((n + m) // 2 - i + 1) * ((n - m) // 2 - i + 1)
        coef = coef // i
        coef = coef // (n - i + 1)
        a[n - 2 * i] = coef

    return a



def evaluate_poly(n, m, a, r):
    """
	Evaluate the polynom of r, defined by the coefficients a[:].
 	Args:
    n : int. Radial order
    m : int. Azimutal order
    a : ndarray. List of coefficients of the polynom, with a[i] the coeff of r**i
    r : ndarray. Variable of the polynomial
    """
    if n > 1:
        r2 = r * r

    p = a[n]
    for i in range(n - 2, m - 1, -2):
        p = p * r2 + a[i]

    if (m == 0):
        return p
    elif (m == 1):
        p *= r
    elif (m == 2):
        p *= r2
    else:
        p = p * r**m

    return p




def zer(r, t, i):
    """
    Compute the Zernike mode number i at coordinates (r,t).
    The numerotation i follows the traditional numbering from Noll 76
    frequently used in AO.
    The algorithm uses a recursive function to compute the coefficients
    of the polynom so that there is no need to use factorial of
    large numbers nor log(gamma(x)), and uses x(x(ax+b)+c)+d to
    compute ax^3+bx^2+cx+d in order to reduce roundoff errors and
    improve computation speed.

    Args:
    r : (float or ndarray) Radius
    t : (float or ndarray) Angle
    i : (int) Number of Zernike mode (1=piston, 2/3=tilts, etc)

    Return:
    
    """
    if (i == 1):
        return np.ones_like(r + t)

    # compute radial and azimutal orders n and m, from i
    n = int((-1. + np.sqrt(8 * (i - 1) + 1)) / 2.)
    p = (i - (n * (n + 1)) / 2)
    k = n % 2
    m = int((p + k) / 2) * 2 - k

    # compute the coefficients of the polynom
    a = polyfute(m, n)

    Z = evaluate_poly(n, m, a, r) * np.sqrt(n + 1)
    if (m != 0):
        Z *= np.sqrt(2)
        if (i % 2):
            Z *= np.sin(m * t)
        else:
            Z *= np.cos(m * t)
    return Z
