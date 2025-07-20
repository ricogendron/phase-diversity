#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 19:37:41 CEST 2025

@author: eric gendron



This module creates something that is "pretty close to an ELT pupil". The pupil
details (outer edges of the segments, gaps, support of spider arms) are very
small structures that will end up being severely undersampled due to the very
coarse sampling that is used in the context of phase diversity. Therefore the
ELT pupil has been simplified, it is a dodecagon with an alternation of edges of
different lengths, equivalent to the intersection at 90° of two hexagons H1 and
H2 of different sizes. One outer hexagon (H1) is 18.6252 m in "radius" (series
of triangles), the other (H2) is 18.732 m (series of trapezes). The central
obscuration is of H1 type, 4.846 m. The f/d of the ELT is given for a diameter
of 38.542 m.

Coordinates comply with the "Standard Coordinate Systems and Basic Conventions
ESO-193058 v6 2015.12.03" in the context of the "[x,y] convention" coding.

"""


import numpy as np






def simplified_elt_pupil(pdiam, edgeblur, u, v, version='old' ):
    """
    Compute a simplified version of the ELT pupil : a kind of dodecagon with 2
    alternative edges at different distances, and an hexagonal obscuration.
    One outer hexagon (H1) is 18.6252 m radius (series of triangles), the other
    (H2) is 18.732 m (series of trapezes). The central obscuration is of H1
    type, 4.846 m radius. The f/d of the ELT is given for a diameter of
    38.542 m.

    Coordinates comply with the "Standard Coordinate Systems and Basic
    Conventions ESO-193058 v6 2015.12.03" in the context of the "[x,y]
    convention" coding.

    Args:
        pdiam (float)   : pupil diameter in pixels corresponding to 38.542m
                          in M1 space
        angle (float)   : pupil angle [rad]
        edgeblur (float): edge blur in percent
        u (2d ndarray)  : coordinates in pixels (possibly rotated)
        v (2d ndarray)  : idem

    Returns:
        2d ndarray : image of the pupil function.
    """
    # Size of diameters, hexagons, in meters, in M1 space.
    D = 38.542 # telescope diameter where the f/d is given
    H1 = 18.6252 # triangle-shaped side ^^^^^^
    H2 = 18.732 # trapeze-shaped side 
    HO = 4.846
    # Rescale above numbers in pixels ("D" --> "pdiam")
    H1 = H1 * pdiam / D
    H2 = H2 * pdiam / D
    HO = HO * pdiam / D
    blur = pdiam * edgeblur/100.

    # let's do the outer part ...............
    relief = np.full_like(u, 2*pdiam)
    for i in range(12):
        a = 2*np.pi*i/12
        relief = np.minimum( [H1,H2][i%2] - (np.cos(a)*u + np.sin(a)*v), relief)
    if blur==0:
        pup_outer = (relief>0).astype(float)
    else:
        pup_outer = np.clip(relief+blur/2,0,blur)/blur
    
    # let's do the obscuration ...............
    relief = np.full_like(u, 2*pdiam)
    for i in range(6):
        a = 2*np.pi*i/6
        relief = np.minimum( HO - (np.cos(a)*u + np.sin(a)*v), relief)
    if blur==0:
        pup_obs = (relief>0).astype(float)
    else:
        pup_obs = np.clip(relief+blur/2,0,blur)/blur


    # Spiders ............... 
    # Algo: we could have drawn 3 spiders that span the full diameter, instead
    # of 6 spiders limited to the radius. But although unnecessary today, this
    # is done on purpose to anticipate the future with 1 spider different from
    # the others.
    nspider = 6
    spiderAngle = np.pi/6
    if version=='old':
        spiderArms = np.full(6, pdiam*0.54/D) # for the time being ...
    else:
        spiderArms = np.array([0.3, 0.54, 0.3, 0.3, 0.3, 0.3]) * pdiam/D # one day, we'll switch
    relief = np.full_like(u, 2*pdiam)
    for i in range(nspider):
        arm_angle = spiderAngle + i*2*np.pi/nspider
        cc, ss = (np.cos(arm_angle), np.sin(arm_angle))
        reliefSpiderLeg = (np.abs(u * ss - v * cc) - spiderArms[i]/2.0)
        no_spider_zone = (u * cc + v * ss) < 0  # identify the right pupil half where the spider arm is
        reliefSpiderLeg[no_spider_zone] = 42. # set to any number greater than 1.0
        relief = np.minimum(relief, reliefSpiderLeg)
    if blur==0:
        pup_spider = (relief>0).astype(float)
    else:
        pup_spider = np.clip(relief+blur/2,0,blur)/blur

    pup_elt = (pup_outer - pup_obs) * (pup_spider)
    return pup_elt


