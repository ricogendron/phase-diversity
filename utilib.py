
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


def line(msg, val, *args, length=60, getval=False):
    """
    Print a nice line on the screen made of 4 items: a message, a series of
    dots, a value, a unit (optional). Example:
    Wavelength ............................. 1653.23 nm
    When getval==True the procedure returns the string instead of printing it.
    
    Parameters
    ----------
    msg    : str, message to be printed
    val    : any type. Value to be printed
    *args  : str, unit (optional argument)
    length : int, optional. Length of the line to print. The default is 60.
    getval : bool, optional. On True, returns the whole string instead of
             printing it. The default is False.
    Returns
    -------
    chain : str. String of the whole line that is to be printed
    """
    n = min(len(msg), 50)  # 60 first caracters
    cutmsg = msg[0:n]
    dots = np.maximum(0, length-n-2) * '.'
    if len(args)>0:
        unit = args[0]
    else:
        unit = ''
    if np.isscalar(val):
        if type(val)==str:
            chain = '%s %s %s %s' % (cutmsg, dots, val, unit)
        else:
            chain = '%s %s %g %s' % (cutmsg, dots, val, unit)
    else:
        valstr = str(val)
        chain = '%s %s %s %s' % (cutmsg, dots, valstr, unit)
    if getval==True:
        return chain
    else:
        print( chain )
 
    