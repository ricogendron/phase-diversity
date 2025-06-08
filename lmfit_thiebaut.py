#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 21:56:25 2022

@author: E Gendron (orig. pgm from Eric Thiebaut)

This program is a brutal translation in python of the original
routine from Eric Thiebaut written in Yorick, lmfit.i (version 2002).

"""


import numpy as np


""" DOCUMENT lmfit -- Non-linear least-squares fit by Levenberg-Marquardt
                     method.

   DESCRIPTION:
     Implement Levenberg-Marquardt method  to  perform  a  non-linear least
     squares fit to a function of an arbitrary number of  parameters.   The
     function  may  be  any  non-linear  function.   If  available, partial
     derivatives can be calculated by the user function, else  this routine
     will  estimate  partial   derivatives   with   a   forward  difference
     approximation.

   CATEGORY:
     E2 - Curve and Surface Fitting.

   SYNTAX:
     result= lmfit(f, x, a, y, w, ...);

   INPUTS:
     F:  The model function  to  fit.   The  function  must  be  written as
         described under RESTRICTIONS, below.
     X:  Anything useful for the model function, for  instance: independent
         variables, a complex structure of  data  or  even  nothing!.   The
         LMFIT routine does not manipulate or use values  in  X,  it simply
         passes X to the user-written function F.
     A:  A vector that contains the initial estimate for each parameter.
     Y:  Array of dependent variables (i.e., the  data).   Y  can  have any
         geometry, but it must be the same as the result returned by F.
     W:  Optional weight,  must be conformable  with Y and all  values of W
         must be positive  or null (default = 1.0).   Data points with zero
         weight are not fitted. Here are some examples:
           - For no weighting (lest square fit): W = 1.0
           - For instrumental weighting: W(i) = 1.0/Y(i)
           - Gaussian noise: W(i) = 1.0/Var(Y(i))

   OUTPUTS:
     A:  The vector of fitted parameters.
     Returns a structure lmfit_result with fields:
       NEVAL:  (long) number of model function evaluations.
       NITER:  (long) number of iteration, i.e. successful CHI2 reductions.
       NFIT:   (long) number of fitted parameters.
       NFREE:  (long) number of degrees of freedom  (i.e.,  number of valid
               data points minus number of fitted parameters).
       MONTE_CARLO: (long) number of Monte Carlo simulations.
       CHI2_FIRST: (double) starting error value: CHI2=sum(W*(F(X,A)-Y)^2).
       CHI2_LAST: (double) last best error value: CHI2=sum(W*(F(X,A)-Y)^2).
       CONV:   (double) relative variation of CHI2.
       SIGMA:  (double) estimated uniform standard deviation of data.  If a
               weight is provided, a  value  of  SIGMA  different  from one
               indicates  that,  if  the  model  is  correct,  W  should be
               multiplied    by     1/SIGMA^2.      Computed     so    that
               sum(W*(F(X,A)-Y)^2)/SIGMA^2=NFREE.
       LAMBDA: (double) last value of LAMBDA.
       STDEV:  (pointer) standard deviation vector of the parameters.
       STDEV_MONTE_CARLO: (pointer)  standard   deviation  vector  of  the
               parameters estimated by Monte Carlo simulations.
       CORREL: (pointer) correlation matrice of the parameters.

   KEYWORDS:
     FIT: List  of  indices  of  parameters  to  fit,  the  others  remaing
          constant.  The default is to tune all parameters.
     CORREL: If  set  to a  non  zero and  non-nil  value, the  correlation
          matrice of the parameters is stored into LMFIT result.
     STDEV: If set to a non zero and non-nil value, the standard deviation
          vector of the parameters is stored into LMFIT result.
     DERIV: When set to a non zero and non-nil  value,  indicates  that the
          model function F is able to compute its derivatives  with respect
          to  the  parameters   (see   RESTRICTIONS).    By   default,  the
          derivatives will be estimated by LMFIT using  forward difference.
          If analytical derivatives are  available  they  should  always be
          used.
     EPS: Small positive value  used  to  estimate  derivatives  by forward
          difference.  Must be such that 1.0+EPS  and  1.0  are numerically
          different  and   should   be   about  sqrt(machine_precision)/100
      (default = 1e-6).
     TOL: Stop criteria for the convergence (default =  1e-7).   Should not
          be smaller  than  sqrt(machine_precision).   The  routine returns
          when  the  relative  decrease of  CHI2 is  less  than  TOL  in an
          interation.
     ITMAX: Maximum number of iterations. Default = 100.
     GAIN: Gain factor for tuning LAMBDA (default = 10.0).
     LAMBDA: Starting value for parameter LAMBDA (default = 1.0e-3).
     MONTE_CARLO: Number of Monte Carlo  simulations to perform to estimate
          standard  deviation  of  parameters  (by default  no Monte  Carlo
      simulations are undergone).  May spend a lot of time if you use a
      large number; but should not be too small!

   GLOBAL VARIABLES:
     None.

   SIDE EFFECTS:
     The values of the vector of parameters A are modified.

   PROCEDURE:
     The function to be fitted must be defined as follow:

       func F(x, a) {....}

     and returns a model with same shape as data Y.  If you want to provide
     analytic derivatives, F should be defined as:

       func F(x, a, &grad, deriv=)
       {
           y= ...;
       if (deriv) {
           grad= ...;
       }
       return y;
       }

     Where X are the independent variables (anything the function  needs to
     compute synthetic data except the model parameters), A  are  the model
     parameters, DERIV is  a  flag  set  to  non-nil  and  non-zero  if the
     gradient is needed and the output gradient GRAD  is  a  numberof(Y) by
     numberof(A) array: GRAD(i,j) = derivative of ith data point model with
     respect to jth parameter.

     LMFIT tune  parameters A so as to minimize:  CHI2=sum(W*(F(X,A)-Y)^2).
     The  Levenberg-Marquardt  method   consists  in  varying  between  the
     inverse-Hessian  method  and the  steepest  descent  method  where the
     quadratic  expansion of  CHI2  does  not  yield  a better  model.  The
     initial  guess of  the parameter  values  should  be as  close to  the
     actual values as possible or the solution may not converge or may give
     a wrong answer.

   RESTRICTIONS:
     Beware that  the result  does depend on your  initial guess A.  In the
     case of  numerous  local  minima,  the only  way to  get  the  correct
     solution is to start with A close enough to this solution.
     
     The estimates of  standard  deviation of the  parameters are  rescaled
     assuming that, for a correct model  and weights, the expected value of
     CHI2 should  be of the  order of NFREE=numberof(Y)-numberof(A)  (LMFIT
     actually compute NFREE from the number of valid data points and number
     of fitted parameters).   If you don't like this you'll have to rescale
     the returned  standard  deviation  to meet  your needs  (all necessary
     information are in the structure returned by LMFIT).

   EXAMPLE:
     This example is from ODRPACK (version 2.01).  The function to fit is
     of the form:
       f(x) = a1+a2*(exp(a3*x)-1.0)^2
     Starting guess:
       a= [1500.0,  -50.0,   -0.1];
     Independent variables:
       x= [   0.0,    0.0,    5.0,    7.0,    7.5,   10.0,
             16.0,   26.0,   30.0,   34.0,   34.5,  100.0];
     Data:
       y= [1265.0, 1263.6, 1258.0, 1254.0, 1253.0, 1249.8,
           1237.0, 1218.0, 1220.6, 1213.8, 1215.5, 1212.0];
     Function definition (without any optimization):
       func foo(x, a, &grad, deriv=)
       {
           if (deriv)
           grad= [array(1.0, dimsof(y)),
                  (exp(a(3)*x)-1.0)^2,
                  2.0*a(2)*x*exp(a(3)*x)*(exp(a(3)*x)-1.0)];
           return a(1)+a(2)*(exp(a(3)*x)-1.0)^2;
       }
       
     Fitting this model by:
       r= lmfit(foo, x, a, y, 1., deriv=1, stdev=1, monte_carlo=500, correl=1)
     produces typically the following result:
        a                   = [1264.84, -54.9987, -0.0829835]
        r.neval             = 12
        r.niter             = 6
        r.nfit              = 3
        r.nfree             = 9
        r.monte_carlo       = 500
        r.chi2_first        = 40.4383
        r.chi2_last         = 40.4383
        r.conv              = 3.84967e-09
        r.sigma             = 0.471764
        r.lambda            = 1e-09
       *r.stdev             = [1.23727, 1.78309, 0.00575123]
       *r.stdev_monte_carlo = [1.20222, 1.76120, 0.00494790]
       *r.correl            = [[ 1.000, -0.418, -0.574],
                               [-0.418,  1.000, -0.340],
                   [-0.574, -0.340,  1.000]]

   HISTORY:
     - Basic ideas borrowed from "Numerical Recipes in C", CURVEFIT.PRO (an
       IDL version by DMS, RSI, of the routine "CURFIT: least squares fit to
       a non-linear function", Bevington, Data Reduction and Error Analysis
       for the Physical Sciences) and ODRPACK ("Software for Weigthed
       Orthogonal Distance Regression" freely available at: www.netlib.org).
     - Added: fitting of a subset of the parameters, Monte-Carlo
       simulations...
"""

def lmfit(fmodel, x, coeffs, data, w=None, fit=None, correl=None, stdev=0, gain=10, tol=1e-7,
          deriv=False, itmax=100, llambda=1e-3, eps=1e-6, monte_carlo=False, verbose=False):
    # Formatting of input data
    a = np.array(coeffs).astype(float)
    na = a.size
    y = data.flatten()
    
    # Maybe we just fit a subset of parameters
    if fit is None:
        fit = np.arange(na)
    elif np.isscalar(fit):
        fit = np.full(1, fit, dtype=int)
    fit = np.array(fit)
    nfit = fit.size
    if nfit<1:
        raise RuntimeError("no parameter to fit.")
    
    # Check weights.
    if w is None:
        w = np.ones_like(y)
    w = np.array(w).flatten()
    if any(w < 0):
        raise RuntimeError("can't handle negative weights.")
    if w.size != y.size:
        raise RuntimeError("dimension of weights not conformable with data.")
        
    nfree = np.sum(w != 0.0) - nfit  # Degrees of freedom
    if nfree <= 0:
        raise RuntimeError("Not enough data points wrt fitted parameters.")

    if 1.0+eps <= 1.0:
        raise RuntimeError(f"Value of EPS is too small ({eps}).")

    warn_zero = True
    warn = "*** Warning: LMFIT "
    neval = 0
    conv = 0.0
    niter = 0
    
    while True:
        if deriv==True:
            m, grad = fmodel(x, a, deriv=True)
            neval += 1
            # Yorick line was : grad= nfit == na ? grad(*,) : grad(*,fit);
            if nfit==na:
                grad = grad[..., :]  # equivalent to yorick's grad(*,) 
            else:
                grad = grad[..., fit]
        else:
            if niter==0:
                m = fmodel(x, a) # first function evaluation
                neval += 1
                grad = np.empty((y.size, nfit), dtype=float) # allocate space
                inc = eps * np.abs(a[fit])
            inc[inc<=0] = eps
            for i in range(nfit):
                anew = a.copy()    # Copy current parameters
                j = fit[i]
                anew[j] = a[j] + inc[i]
                grad[:,i] = (fmodel(x, anew)-m).flatten() / inc[i]
            neval += nfit
        
        chi2 = y - m # difference with the data
        beta = w * chi2
        if niter>0:
            chi2 = chi2new
        else:
            chi2_first = np.sum(beta * chi2)
            chi2 = chi2_first
        beta = beta.flatten().dot(grad)  # grad(+,) * beta(*)(+);
        alpha = grad.T.dot( w[:,None] * grad ) # (w(*)(,-) * grad)(+,) * grad(+,);
        gamma = np.sqrt( np.diag(alpha) )
        if any(gamma <= 0.0):
            # Some derivatives are null (certainly because of rounding errors).
            if warn_zero :
                print(warn, "Zero derivatives were found.")
                warn_zero = False
            gamma[gamma <= 0.0] = eps * np.max(gamma)
            # goto done
        
        gamma = 1.0 / gamma
        beta *= gamma
        alpha *= gamma[:, None] * gamma[None, :] # gamma(,-) * gamma(-,);

        while True:
            for i in range(len(alpha)):
                alpha[i,i] = 1.0 + llambda # alpha[diag] = 1.0 + llambda;
            anew = a.copy()
            anew[fit] += gamma * np.linalg.solve(alpha, beta)
            m = fmodel(x, anew)
            neval +=1
            d = y-m
            chi2new = np.sum(w*d*d)
            if chi2new < chi2:
                break
            llambda *= gain
            if all( anew==a ):
                # No change in parameters. 
                print(warn, "making no progress.")
                return a
        
        a = anew.copy()
        llambda /= gain
        niter += 1
        conv = 2.0*(chi2-chi2new)/(chi2+chi2new)
        if verbose==True:
            print(f'Iteration: {niter:3d}     Chi2: {chi2new/nfree:}     Progress: {conv:.6f}  ')

        if conv <= tol:
            break
        if niter >= itmax:
            print(warn + "reached maximum number of iterations (%d)." % itmax)
            break
    return a



