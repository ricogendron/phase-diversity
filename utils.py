
import numpy as np



def regress(y, x):
    """
    Compute the linear regression coefficient A between y and x

    Args:
        y (ndarray): 1d data
        x (ndarray): 1d data

    Returns:
        float: linear regression coefficient
    """
    xm = np.mean(x)
    a = (np.mean(y*x) - np.mean(y)*xm) / (np.mean(x*x) - xm*xm)
    return a



def grint(msg):
    """
    Print msg in green.

    Args:
        msg (str): Message to print.
    """
    print("\033[32m" + msg + "\033[0m")

def rrint(msg):
    """
    Print msg in red.

    Args:
        msg (str): Message to print.
    """
    print("\033[31m" + msg + "\033[0m")

def brint(msg):
    """
    Print msg in blue.

    Args:
        msg (str): Message to print.
    """
    print("\033[34m" + msg + "\033[0m")