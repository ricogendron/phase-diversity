#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 18:08:47 2025

@author: eric gendron

"""

import numpy as np
import zernike as zer
from lmfit_thiebaut import lmfit
from utils import regress, grint, rrint, line

import matplotlib.pyplot as plt
plt.ion() # interactive mode on, no need for plt.show() any more



def make_modal_basis(x, y, defoc):
    """Author: EG
    Compute a modal basis defined on the useful pixels of the pupil with the
    native sampling. The basis is ordered by increasing spatial frequencies, it
    is made of orthogonal modes. The three first modes B[:,0:3] are forced to be
    pure tip, tilt and defocus. All the modes are orthogonal to piston.

    Args:
        x (ndarray): x coordinate of all the pupil pixels, normalized to 1. at
                     the edge of the pupil main axis.
        y (ndarray): idem in y
        defoc (ndarray): idem for r^2

    Returns:
        2d ndarray: matrix of the column-vectors of the modes
    """
    mat = ((x[:,None]-x[None,:])**2 + (y[:,None]-y[None,:])**2)**(5/6)
    nphi = x.size
    P = np.eye(nphi) - np.ones((nphi,nphi))/nphi # piston-removal matrix
    mat = P @ mat @ P
    # diagonalise the matrix. The eigenvalues are sorted in increasing order
    s_eig, B = np.linalg.eigh(mat)

    # Here we already have nice modes. But now we will remove the
    # tip-tilt-defoc contribution from all the modes, and put tiptiltdefoc
    # at the beginning of the basis.
    defocop = defoc - np.mean(defoc) # orthogonalize wrt piston
    ttmat = np.array([x, y, defocop]).T  # tip-tilt-defoc matrix
    ttproj = np.linalg.pinv(ttmat) # projection matrix on tip-tilt-defoc
    ttcomp = ttproj.dot(B) # tiptilt-defoc contained in every mode
    B = B - ttmat.dot(ttcomp) # tiptilt-defoc removal from all modes
    # determine where were initially the closest modes resembling to tip, tilt, defoc in the basis
    ttd_index = np.argmax(np.abs(ttcomp), axis=1)
    # determine the index where all the other modes were
    tmp = np.ones_like(x, dtype=bool)
    tmp[ttd_index] = False
    (others_index,) = np.nonzero(tmp)
    # put tiptilt-defoc back in B at the beginning, followed by all the others
    B = np.concatenate((ttmat, B[:,others_index]), axis=1)
    # Let's normalise the modes
    B = B / np.sqrt((B**2).sum(axis=0))[None,:]
    return B


def fcrop(img, Ncrop):
    """Author: EG
    Crop a series of images of shape (3,N,N) to a size (3,Ncrop,Ncrop) only if
    Ncrop<N, otherwise do nothing. Also do a fftshift before and after cropping,
    as the crop is intended around the pixel [0,0]. This function is useful for
    reformatting the large images of size N from the computations, to the small
    data image format.

    Args:
        img (ndarray): image cube of shape (nbim,N,N)
        Ncrop (int): size the images shall be cropped to

    Returns:
        ndarray: image cube of shape (nbim,Ncrop,Ncrop)
    """
    nbim, N, _ = img.shape
    if Ncrop<N:
        dN = N-Ncrop
        dn = dN//2
        tmp = np.fft.fftshift(img, axes=(1,2))[:, dn:dn+Ncrop, dn:dn+Ncrop]
        tmp =  np.fft.fftshift(tmp, axes=(1,2))
        return tmp
    else:
        return img


def check_image_format(img, xc, yc, N):
    """Author: EG
    Convert a list of images to a cube with defocus in the first dimension.
    If the images are not square they will be cut to the largest square format
    that fits in the original rectangle image.
    If the parameters xc and yc have been provided (not None) the images will
    be cropped to the largest square format centred on [xc,yc] that fits in the
    original image.
    If the parameter N is provided the square format (N,N) will be forced.

    Args:
        img (ndarray)  : either a list of images or a 3D cube with defocus in the first dimension
        xc (int | None): index of the center of the image, along X. If None, the center of the
                         image will be used
        yc (int | None): idem along Y.
        N (int | None) : size of the square image. If None, the largest square format that fits in the
                         original image will be used. If N is odd, it will be forced to even.
                         If N is not None, xc and yc are ignored.

    Returns:
        ndarray: image cube with the shape (ndefoc, N, N).
    """    
    print_user_manual = False
    # if it's a list, convert to ndarray cube
    if type(img)==list:
        img = np.array(img)

    # verify that it's an ndarray cube with defocus in 1st dimension
    if img.ndim==3:
        kdefoc, ix, iy = img.shape # the 1st dim must be the smallest dimension (usually < 5)
        if kdefoc!=np.min([kdefoc, ix, iy]):
            print_user_manual = True
    else:
        print_user_manual = True

    if print_user_manual == True:
        raise ValueError(
            """
            Data format cannot be handled. Please provide a data cube with a shape
            of the form (ndefoc, N, N), or a list of ndefoc images with a size (N, N).
            If the images are not square they will be cut to the largest square format
            that fits in the original rectangle image.
            If the parameters xc and yc have been provided (not None) the images will
            be cropped to the largest square format centred on [xc,yc] that fits in the
            original image.
            If the parameter N is provided the square format (N,N) will be forced.
            """)
    # crop to a square format, centred on (xc, yc)
    if xc is None:
        xc = ix//2
    if yc is None:
        yc = iy//2
    Ncrop = 2*np.min(np.array([xc,ix-xc,yc,iy-yc]))
    if N is not None:
        Ncrop = np.minimum(Ncrop, N)
        Ncrop = Ncrop - Ncrop%2 # force even
    n = Ncrop//2
    img = img[:, xc-n:xc+n, yc-n:yc+n]
    img = np.fft.fftshift(img, axes=(1,2))
    return img





class Opticsetup():
    def __init__(self, img_collection, xc, yc, N, defoc_z,
                 pupilType, flattening, obscuration, angle, nedges,
                 spiderAngle, spiderArms, spiderOffset, illum,
                 wvl, fratio, pixelSize, edgeblur_percent,
                 object_fwhm_pix, object_shape='gaussian'):
        """Author: EG
        Creation of the Opticsetup class.

        Args:
            img_collection (ndarray | list): data cube (ndefoc, N, N), or a list
                                of images with a size (N, N) obtained experimentally.
            xc (int | None)   : x coordinate of the center of the image. If None, the center of the
                                image will be used.
            yc (int | None)   : y coordinate of the center of the image. If None, the center of the
                                image will be used.
            N (int | None)    : size of the square image. If None, the largest square format that fits
                                in the original image will be used. If N is odd, it will be forced to
                                be even.
            defoc_z (ndarray | list): list of the amount of defocus in [m] for each image.
            pupilType (int):    0: disk/ellipse, 1: polygon, 2: ELT
            flattening (float): flattening factor of the pupil. Applies to circular (disk) apertures
                                as well as to polygonal apertures. The flattening operates in a direction
                                perpendicular to the angle of the pupil. The flattening is defined as
                                the ratio of the minor axis to the major axis of the ellipse.
            obscuration (float):obscuration factor of the pupil. Applies to circular (disk) apertures
                                as well as to polygonal apertures.
            angle (float)     : angle of the pupil in radians. 
            nedges (int)      : number of edges of the polygon pupil. Only useful for polygon pupil.
            spiderAngle (float):angle of the spider in radians, counted from the axis defined by the
                                angle of the pupil. 
            spiderArms (list) : list of the widths of each spider arms in [relative units]. The number
                                of arms is
                                equal to the number of elements in this list. An empty list [] means
                                no spider. The arms are assumed to be straight lines.
            spiderOffset (list):list of the offsets the line of each spider arm wrt the centre of the
                                pupil, in [relative units]. The length of the list must be equal to the
                                length of the spiderArms list. 
            illum (list | ndarray): illumination coefficients of the pupil. The illumination map is
                                described using Zernike coefficients. The first coefficient is the
                                piston term, Z_1(r,t) = 1.00, and correspond to a flat illumination.
                                Therefore for a simple, flat illumination, the list should just be [1.0].
                                The list can be of any arbitrary length.
            wvl (float): wavelength of the light in [m]. 
            fratio (float): focal ratio of the setup forming the images of the data cube.
            pixelSize (float): size of the pixel in [m].
            edgeblur_percent (float): percentage of the edge blur applied to the edges of the pupil.
            object_fwhm_pix (float): FWHM of the object in [pixels]. The object is assumed to be either
                                a Gaussian or a disk. A value of 0.0 means an infinitely small object.
            object_shape (str) : Shape of the object, either 'gaussian' or 'disk' or 'square'
        """
        # format the images and return a cube
        self.img = check_image_format(img_collection, xc, yc, N) # list of images
        # dimensions of the data cube: nbim=number of defocused images, Ncrop = size of data
        self.nbim, self.Ncrop, _ = self.img.shape # number of images
        # self.N is the image size for the computations
        if N is None:
            self.N = self.Ncrop
        else:
            self.N = N
        line('Data image format', f'{self.Ncrop}x{self.Ncrop}')
        line('Computation format', f'{self.N}x{self.N}')

        if len(defoc_z)==self.nbim:
            self.defoc_z = np.array(defoc_z)
        else:
            raise ValueError('Number of images ({self.nbim}) does not match number of defocus ({len(defoc_z)}).')
        
        self.focscale = 1.0

        self.pupilType = ['disk','polygon','ELT'][pupilType] # self.pupilType = ['disk','polygon','VLT','ELT','GMT'][pupilType]  ... one beautiful day
        self.flattening = flattening
        if obscuration<1.0:
            self.obscuration = obscuration
        else:
            raise ValueError(f'Value of the obscuration is expressed relative to the pupil diameter. It must be in the interval [0, 1[ (not {obscuration}).')
        self.angle = angle
        self.nedges = nedges # only useful for polygon pupil
        
        self.spiderAngle = spiderAngle
        self.spiderArms = spiderArms
        self.spiderOffset = spiderOffset
        self.nspider = len(self.spiderArms)
        self.edgeblur = np.maximum(edgeblur_percent, 1e-6) # in percent

        self.illum = illum # zernike list of illum, starting with piston

        self.wvl = wvl # wavelength in [m]

        self.fratio = fratio # ratio f/D
        self.pixelSize = pixelSize # size of pixels in [m]

        # Evaluate the number of pixels required in the pupil diameter to
        # match the plate scale of the images
        self.pdiam = self.compute_pupil_diam()
        line(f'Nb of phase pixels pupil diam (f/{self.fratio:6.3f})', self.pdiam, 'pixels')
        line('Angular size of a phase pixel in pupil plane', f'f/{self.fratio*self.pdiam:7.3f}')

        # standard variables
        # create variables x, y over whole square support
        tmp = np.arange(self.N) - self.N/2.0
        self.x, self.y = np.meshgrid(tmp, tmp, indexing='ij')

        # pupil model
        self.pupilmap = self.pupilModel()
        self.idx = self.pupilmap.nonzero() # index of pixels of the useful pupil part

        # standard variables defined only over the pupil area for zernike evaluation
        rdiam = self.pdiam/2 # pupil radius in pixels
        self.tip = 2 * self.x[self.idx] / rdiam
        self.tilt = 2 * self.y[self.idx] / rdiam
        self.r = np.sqrt(self.x[self.idx]**2 + self.y[self.idx]**2) / rdiam
        self.theta = np.arctan2(self.y[self.idx], self.x[self.idx])
        self.defoc = zer.zer(self.r, self.theta, 4)
        # conversion coefficients from [radians rms of tilt Zernike] to some
        # displacement in the focal plane
        self.rad2dist = 4 * self.fratio * (self.wvl / 2 / np.pi) # convert fact from a2[rad] to metres
        self.rad2pix = self.rad2dist / self.pixelSize # convert fact from a2[rad] to pixels
        # conversion coefficient from [radians rms of Zernike defocus] to some
        # displacement along the optical axis
        self.rad2z = (self.wvl / (2*np.pi)) * (16 * np.sqrt(3) * self.fratio**2)

        # number of phase points that will be treated.
        nphi = self.idx[0].size
        line("Number of phase points in the pupil", nphi)
        # Here we are going to define a new basis for the phase, that will be obtained
        # by diagonalisation of the matrix of the pairwise distances**(5/3) between the useful
        # pixels of the pupil.
        self.phase_basis = make_modal_basis(self.tip, self.tilt, self.defoc) # 3 first modes are tip, tilt and defoc
        self.phase = np.zeros(10)

        # The modes are normalized in a weird way, that depends on the shape of
        # the pupil. Below we search the normalisation factor with the zernike
        # modes in order to be able to retrieve knwon units.
        # self.convert * self.phase[0:3] --> the coefficients of Zernike modes
        self.convert = np.array([regress(self.phase_basis[:,0], self.tip),
                        regress(self.phase_basis[:,1], self.tilt),
                        regress(self.phase_basis[:,2], self.defoc)])

        # calculate the pupil illumination using the Zernike polynomials just over the pupil area
        self.pupillum = self.compute_illumination(self.illum)

        self.wrap = []
        self.weight = None

        # define some useful variables that will be used later in the lmfit
        self.optax_x = np.zeros(self.nbim)
        self.optax_y = np.zeros(self.nbim)
        self.amplitude = np.ones(self.nbim)
        self.background = np.zeros(self.nbim)

        # compute the Fourier transform of the square shape covered by a single
        # pixel. When the FWHM is zero the function compute_tf_object returns
        # the scalar 1.0, to fasten the computation.
        self.tf_pixobj = self.compute_tf_object(1.0, 'square') # pixel MTF

        # Later on, this will be multiplied by the Fourier transform of an
        # object with a given FWHM. 
        self.object_shape = object_shape
        self.object_fwhm_pix = object_fwhm_pix




    def compute_tf_object(self, object_fwhm_pix, type='gaussian'):
        """Author: EG
        Compute the Fourier transform of an object, with a given FWHM.
        The object FWHM is given in [pixels].
        The object is assumed to be either a Gaussian, a disk or a square.

        Args:
            object_fwhm_pix (float): FWHM of the object in [pixels].
            type (str, optional)   : type of object. Defaults to 'gaussian'.
                                    'gaussian' : Gaussian object
                                    'disk'    : disk object (fwhm is diameter)
                                    'square'  : square object (fwhm is side length)
        Returns:
            ndarray: Fourier transform of the object.
        """
        if object_fwhm_pix<=0:
            return 1.0
        if type=='gaussian':
            # The Fourier transform of a Gaussian is another Gaussian
            a = object_fwhm_pix / 1.66511 # suitable for the fwhm of exp(-(x/a)**2)
            a = self.N/(np.pi*a)
            object_tf = np.exp((self.x**2 + self.y**2)/(-a**2))
            object_tf = np.fft.fftshift(object_tf)
        elif type=='disk':
            # The Fourier transform of a disk is a bessel function 2*J1(x)/x
            from scipy.special import j1
            r = np.sqrt(self.x**2 + self.y**2) * (np.pi*object_fwhm_pix/self.N)
            r = np.fft.fftshift(r)
            r[0,0] = 1.0
            object_tf = 2*j1(r)/r
            object_tf[0,0] = 1.0 # 2*J1(eps)/eps = 1
        elif type=='square':
            # The Fourier transform of a square is a sinc(x)*sinc(y) function
            object_tf = np.sinc(self.x*(object_fwhm_pix/self.N)) * np.sinc(self.y*(object_fwhm_pix/self.N))
            object_tf = np.fft.fftshift(object_tf)
        else:
            raise ValueError(f"Unknown object type: {type}")
        return object_tf


    def phase_generator(self, vector, tiptilt=True, defoc=True):
        """Author: EG
        Convert the vector of modal coefficients to a zonal (=pixel list)
        representation of the phase. The phase is computed using the phase
        basis, which is a set of orthogonal modes defined over the pupil area.
        The phase is computed only over the pixels of the pupil area.

        Args:
            vector (ndarray)  : vector of modal coefficients to be converted to a
                                zonal (=pixel list) representation of the phase.
            tiptilt (bool)    : optional. The tip/tilt value is filtered out when
                                False. Default is True. 
            defoc (bool)      : optional. The defoc value is filtered out when
                                False. Default is True. 
        Returns:
            ndarray: list of the phase values in the pupil
        """
        vector = np.array(vector)
        # size of the vector where matrix prod shall be done
        n = np.minimum(vector.size, self.phase_basis[0,:].size)
        # treatment of tiptilt and defoc flags
        idx = np.ones(n, dtype=bool)
        idx[0:2] = tiptilt
        idx[2] = defoc
        (idx,) = np.nonzero(idx)
        # convert the modal vector to a list of phase values
        phase = self.phase_basis[:,idx] @ vector[idx]
        return phase


    def compute_illumination(self, zernik_illum_array):
        """Author: EG
        Compute the amplitude of the pupil illumination, using Zernike
        polynomials weighted by a set of coefficients.

        Args:
            zernik_illum_array (list | ndarray): list of the weighting
                        coefficients of the Zernike polynomials used to describe
                        the modulus of the complex amplitude of the pupil
                        illumination. The first coefficient is the piston term,
                        Z_1(r,t) = 1.00, and correspond to a flat illumination.
                        Therefore for a simple, flat illumination, the list
                        should just be [1.0]. The list can be of any arbitrary
                        length.
        Returns:
            1d ndarray: list of values of the pupil illumination in the pupil
                        area.
        """        
        # pupil illumination
        tmp = 0.0
        for (i,coef) in enumerate(zernik_illum_array, start=1):
            tmp = tmp + zer.zer(self.r, self.theta, i)*coef
        pupillum = self.pupilmap[self.idx] * tmp
        return pupillum


    def mappy(self, contents, within=None):
        """
        Transforms a vector of values defined over the pupil area to a 2D map of
        the same size as the image. The values are set to zero outside the pupil
        area. The function also allows to initialize the map with a given array
        (within).

        Args:
            contents (ndarray): vector of values defined over the pupil area.
            within (2d ndarray, optional): array to initialize the map. If None,
            the map is initialized to zero.

        Returns:
            2d ndarray: 2D map of the same size as the image.
        """        
        if within is None:
            var = np.zeros(contents.shape[:-1] + (self.N, self.N), dtype=contents.dtype)
        else:
            var = within.copy().astype(contents.dtype)
        var[...,self.idx[0],self.idx[1]] = contents
        return var


    def compute_pupil_diam(self):
        """
        Compute the best number of pixels in the pupil diameter to match the
        plate scale of the images. It is computed using the formula:
        N * pixelsize / ( f/D * wavelength) 

        Returns:
            float: number of pixels in the pupil diameter.
        """        
        # nb of pixels in airy (= nb of pupils in support)
        ld = self.fratio * self.wvl / self.pixelSize
        # nb of pixels in the pupil diameter
        pdiam = self.N / ld
        return pdiam
    
    
    def pupilModel(self):
        """
        Compute the pupil function. The pupil function is a 2D array of the same
        size as the image. The pupil size is defined by the diameter of the
        pupil, which is computed to match the plate scale of the images after
        the FFT.

        Returns:
            ndarray 2d: image of the pupil function.
        """
        # central obscuration: work-around needed to really discard
        # obscurations with a negative or null diameter<=0.0 
        obs = self.obscuration if self.obscuration>1e-6 else -0.5
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        u = self.x*c + self.y*s
        v = self.x*(-s/self.flattening) + self.y*(c/self.flattening)
        blur = self.edgeblur/100*self.pdiam
        # width and middle of donut
        Am = self.pdiam*(1 + obs)/4 # middle between R and R.robs
        Dm = self.pdiam*(1 - obs)/2 # width of the donut
        if self.pupilType=='disk':
            # distance squared
            r2 = u*u + v*v
            # Compute d=(r-A), and r=0.5-d
            relief = (Dm/2 - np.abs(np.sqrt(r2) - Am))/blur
        elif self.pupilType=='polygon':
            relief = np.zeros_like(u)
            for i in range(self.nedges):
                a = 2*np.pi*i/self.nedges
                relief = np.maximum(np.cos(a)*u + np.sin(a)*v, relief)
            relief = (Dm/2 - np.abs(relief - Am))/blur 
        else:
            relief = np.ones_like(u)

        # Spiders
        for i in range(self.nspider):
            arm_angle = self.spiderAngle + i*2*np.pi/self.nspider
            cc = np.cos(arm_angle)
            ss = np.sin(arm_angle)
            reliefSpiderLeg = (np.abs( u * ss - v * cc + self.spiderOffset[i]*self.pdiam) - self.spiderArms[i]*self.pdiam/2.0) / blur
            no_spider_zone = (u * cc + v * ss) < 0  # identify the right pupil half where the spider arm is
            reliefSpiderLeg[no_spider_zone] = 42. # set to any number greater than 1.0
            relief = np.minimum(relief, reliefSpiderLeg)

        img = np.clip(relief + 0.5, 0, 1)
        return img
    

    def pupilArtist(self, pltax, color='r', linewidth=0.6):
        """
        Don't ask. It's just a first attempt. Not sure it will stay here for long.

        """  
        def rotation(angle,x,y):
            c = np.cos(angle)
            s = np.sin(angle)
            return (c*x - s*y, s*x + c*y)

        uc, vc = (self.N/2, self.N/2)
        if self.pupilType=='disk':
            N = 222
            th = np.linspace(0, 2*np.pi, N)
            x = np.cos(th)*self.pdiam/2
            y = np.sin(th)*self.pdiam*self.flattening/2
            u, v = rotation(self.angle,x , y)
            pltax.plot(u + uc, v + vc, linewidth=linewidth, color=color)
            if self.obscuration>0:
                u = u * self.obscuration
                v = v * self.obscuration
                pltax.plot(u + uc, v + vc, linewidth=linewidth, color=color)
        elif self.pupilType=='polygon':
            th = np.linspace(0, 2*np.pi, self.nedges+1, endpoint=True) + np.pi/self.nedges
            x = np.cos(th)*self.pdiam/2/np.cos(np.pi/self.nedges)
            y = np.sin(th)*self.pdiam*self.flattening/2/np.cos(np.pi/self.nedges)
            u, v = rotation(self.angle, x, y)
            pltax.plot(u + uc, v + vc, linewidth=linewidth, color=color)
            if self.obscuration>0:
                u = u * self.obscuration
                v = v * self.obscuration
                pltax.plot(u + uc, v + vc, linewidth=linewidth, color=color)
        else:
            pass

        # Spiders arms
        for i in range(self.nspider):
            arm_angle = self.spiderAngle + i*2*np.pi/self.nspider
            x = np.array([np.maximum(0, self.obscuration*self.pdiam), self.pdiam]) / 2.
            y1 = (self.spiderOffset[i] - self.spiderArms[i]/2.) * self.pdiam * np.ones(2)
            y2 = (self.spiderOffset[i] + self.spiderArms[i]/2.) * self.pdiam * np.ones(2)
            uu, vv = rotation(arm_angle, x, y1)
            u, v = rotation(self.angle, uu, vv * self.flattening)
            pltax.plot(u + uc, v + vc, linewidth=linewidth, color=color)
            uu, vv = rotation(arm_angle, x, y2)
            u, v = rotation(self.angle, uu, vv * self.flattening)
            pltax.plot(u + uc, v + vc, linewidth=linewidth, color=color)



    def psf(self, phase):
        """
        Compute the series of the defocused PSF images based on the
        defocus coefficients and the phase. The PSF is computed using
        the inverse/backward Fourier transform of the pupil function.

        Args:
            phase (ndarray 1d): vector of modal coefficients that define the phase.

        Returns:
            ndarray 3d: cube of PSF images with the shape (nbim, N, N).
        """        
        # convert defocus coeffs from dZ to radians RMS
        defoc_a4 = self.defoc_z / self.rad2z  # in radians RMS
        # compute the various defocus maps (only on useful pixels)
        tmp_defoc = self.defoc[None,:] * defoc_a4[:,None] + self.phase_generator(phase) # shape (nbim, nphi)
        tmp = self.pupillum[None,:]*np.exp(1j*tmp_defoc) # shape (nbim, nphi)
        # transform to a two-dimensional map
        tmp = self.mappy(tmp) # shape (nbim, N, N)
        # allocate the PSF array
        psfs = np.zeros((self.nbim, self.N, self.N)) # shape (nbim, N, N)
        # Loop over the defocus images: compute the PSF for each defocus, then convolution
        # of the psf with the pixel area and with an object, if any. The fourier transform
        # of the object is already computed in the Opticsetup class (tf_pixobj)
        for i in range(self.nbim):
            localpsf = (self.amplitude[i] * np.abs(np.fft.ifft2( tmp[i,:,:] )))**2
            psfs[i,:,:] = np.fft.fft2(np.fft.ifft2(localpsf) * self.tf_pixobj).real + self.background[i]
        return psfs
    

    def encode_coefficients(self, *args):
        """
        Concatenate all the coefficients of the input arguments into a single
        array. The function accepts any number of arguments, which can be
        scalars, lists or ndarrays. The function returns a 1D array containing
        all the coefficients. The function also keeps track of the type of each
        input argument (scalar, list, ndarray) and its shape. This information
        is stored in the self.wrap attribute. The self.wrap attribute is a list
        of tuples, where each tuple contains the type of the input argument, its
        length and its shape. This information is used later to decode the
        coefficients.

        Args:
            any number, any ndarray, list of floats or scalar values:
            coefficients to be encoded.

        Returns:
            ndarray 1d: all coefficients concatenated in a single array.
        """        
        itemlist = []
        self.wrap = []
        for i, item in enumerate(args):
            # if scalar value, then make an array
            if np.isscalar(item):
                self.wrap.append(('scalar', 1, (0,)))
                tmp = np.full(1, item)
            elif type(item) == list:
                self.wrap.append(('list', len(item), (0,)))
                tmp = np.array(item)
            elif isinstance(item, np.ndarray):  # Check if it's a NumPy array
                if item.ndim > 1:
                    self.wrap.append(('ndarray', item.size, item.shape))
                    tmp = item.flatten()
                elif item.ndim == 0:
                    raise RuntimeError(f'The item n°{i} is invalid, it has 0 dimension.')
                elif item.ndim == 1:
                    self.wrap.append(('1darray', item.size, item.shape))
                    tmp = item
            else:
                raise RuntimeError(f'The item n°{i} is invalid: \n{item}')
            # append it to the list
            itemlist.append(tmp)
        itemlist = np.concatenate(itemlist)
        return itemlist

        
    def decode_coefficients(self, itemlist):
        """
        Symmetric function of the encode_coefficients function. The function
        takes a 1D array of coefficients and decodes it into a tuple containing
        the original input arguments. The function uses the self.wrap attribute
        to determine the type and shape of each input argument.

        Args:
            itemlist (ndarray 1d): 1D array of coefficients to be decoded.
        Returns:
            tuple: tuple containing the original input arguments.
        """        
        ptr = 0
        output = () # prepare tuple output to receive other params
        # 'deckey' is the decode-key that allows us to know what kind of parameter
        # had been stored
        for deckey in self.wrap:
            datatype, length, shape = deckey
            if datatype=='scalar':
                output = output + (itemlist[ptr],)
            elif datatype=='list':
                output = output + (itemlist[ptr:ptr+length].tolist(),)
            elif datatype=='ndarray':
                tmp = np.reshape(itemlist[ptr:ptr+length], shape)
                output = output + (tmp,)
            elif datatype=='1darray':
                output = output + (itemlist[ptr:ptr+length],)
            else:
                raise ValueError(f'Decoding keyword {datatype} is unknown. Cant decode the sequence.')
            ptr = ptr+length
        return output



    def manage_fitting_flags(self, defoc_z_flag, focscale_flag,
                            optax_flag, amplitude_flag, background_flag, phase_flag,
                            illum_flag, objsize_flag):
        """
            Returns the list of the indices of the parameters to be fitted among the
        "large encoded array" that is passed to the minimisation function, based on
        all the boolean flags that tell which parameter shall (or not) be fitted as
        the input.
            Moreover the function shall manage the problem of redundancies (i.e.
        parameters that appear twice, or that have identical effects). The redundant
        degrees of freedom (DoF) must be eliminated. Elimination consists in setting
        some of them to False, and the choice is done on a case-by-case basis.

            Clearly, this function should work alongside the 'encode' function,
        given that they are both responsible for constructing the series of
        coefficients and their fitting index. However, I haven't found a good way of
        getting the two functions to work together. The encode() function is rather
        automatic and can cope with any arbitrary input, whereas the
        manage_fitting_flag() function is highly customised and specific due to its
        management of DoF duplication. To be improved in a next version.

        Args:
            osetup (Opticsetup): the optical setup object
            *args (bool or ndarray(bool)): flags for fitting (or not) the parameters

        Returns:
            ndarray: the series of indexes of the coefficients to be fitted.
        """
        # ........ first expand each scalar bool into a proper array with proper length
        defoc_z_flag    = np.full(len(self.defoc_z),    defoc_z_flag)
        focscale_flag   = np.full(1,                    focscale_flag)
        optax_flag      = np.full(len(self.optax_x),    optax_flag)
        amplitude_flag  = np.full(len(self.amplitude),  amplitude_flag)
        background_flag = np.full(len(self.background), background_flag)
        phase_flag      = np.full(len(self.phase),      phase_flag)
        illum_flag      = np.full(len(self.illum),      illum_flag)
        objsize_flag    = np.full(1,                    objsize_flag)
        # ........ manage redundancy conflicts between tip-tilt and position of
        # optical axis: first one count nb of occurences of a 'True' statement
        # for fitting the parameter (n_true), then one compares to the number of
        # degrees of freedom (DoF) available for this parameter and calculate
        # the difference (redundancy), then one applies a fix by imposing some
        # params flags to False.
        n_true = np.count_nonzero(optax_flag) + np.count_nonzero(phase_flag[0:1])
        redundancy = n_true - self.nbim
        if redundancy>0 :
            rrint(f'Redundant configuration (+{redundancy}) found for tip.')
            print(' - Blocking tip DoF on phase vector. Phase will be tip-free.')
            phase_flag[0] = False
        # ........ manage redundancy conflicts between tilt & optical axis position
        n_true = np.count_nonzero(optax_flag) + np.count_nonzero(phase_flag[1:2])
        redundancy = n_true - self.nbim
        if redundancy>0 :
            rrint(f'Redundant configuration (+{redundancy}) found for tilt.')
            print(' - Blocking tilt DoF on phase vector. Phase will be tilt-free.')
            phase_flag[1] = False
        # ........ manage redundancy conflicts for defocus
        n_true = np.count_nonzero(defoc_z_flag) + np.count_nonzero(phase_flag[2:3]) + np.count_nonzero(focscale_flag)
        redundancy = n_true - self.nbim
        if redundancy>0:
            rrint(f'Redundant configuration (+{redundancy}) found for defocus.')
            if phase_flag[2]==True:
                phase_flag[2] = False
                redundancy -= 1 # decrement
                print(' - Blocking defocus DoF in the phase vector. Phase will be defocus-free.')
        if redundancy>0 :
            i = np.argmax(np.abs(self.defoc_z))
            if defoc_z_flag[i]==True:
                defoc_z_flag[i] = False
                redundancy -= 1 # decrement
                print(f' - Blocking defocus DoF on defocus vector at position {i+1}.')
        # ........ manage redundancy conflicts on the amplitude/illumination 
        n_true = np.count_nonzero(amplitude_flag) + np.count_nonzero(illum_flag[0:1])
        redundancy = n_true - self.nbim
        if redundancy>0 :
            rrint(f'Redundant configuration (+{redundancy}) found for the flux on the images.')
            print(' - Blocking illumination DoF in the illumination vector.')
            illum_flag[0] = False
            redundancy -= 1 # decrement

        list_flag = np.concatenate((defoc_z_flag, focscale_flag, optax_flag, optax_flag,
                                    amplitude_flag, background_flag, phase_flag, illum_flag,
                                    objsize_flag))
        # get the indices of the coefficients to be fitted
        (fit_flag,) = list_flag.nonzero()
        return fit_flag



    def zerphase(self, J=21, tiptilt=False, defoc=True):
        """
        Compute the decomposition of the phase (deprived from piston and
        tip-tilt by default) into Zernike coefficients. Boolean flags allow the
        user to either keep or ignore tiptilt and/or defocus in the input phase. 

        Warning: This decomposition only makes full sense when the pupil is
        circular. For other pupil shapes, the coefficients obtained must be
        treated with extreme caution. Zernike modes are defined as usual as
        Zi(r/R, theta), with theta=0 directed along the X-axis (horizontally,
        rather than along the main axis of the pupil). The unitary disk where
        the Zernike modes are defined, is mapped onto the circle of a diameter
        defined by the f-ratio. Then, the real pupil of the system acts as a
        mask over this. The decomposition is performed by de-coupling from the
        non-orthogonality of the polynomials on the support/mask they are
        computed and using the native pupil sampling, in such a way that their
        weighted sum restitutes back the retrieved phase over the system pupil.
        Values outside the system pupil are meaningless (they should not be
        computed). On non-circular pupils, RMS values of the coefficients are
        meaningless. The decomposition includes a piston and tiptilt terms,
        although the phase is piston-free and tilt-free over the pupil aperture.
        This is required because of the non-orthogonality. The piston and tilt
        values induced by the various polynomials need to be annihilated by Z_1
        and Z_2/3 terms.
          As only a limited number of polynomials are involved (21 by default),
        their combination may not fully recreate the initial wavefront.
        Therefore the routine also returns the RMS value of the amount of
        leftovers that were ignored in the expansion, just for information.

        Args:
            J (int, optional) : Number of the last Zernike mode that is
                     considered in the decomposition. Default is 21.
            tiptilt (bool, optional) : flag to keep or ignore the tiptilt in 
                    the input phase. Default is False.
            defoc (bool, optional) : flag to keep/ignore the defocus in the
                    input phase. Default is True.
        Return:
            zervector (1D ndarray) : array of Zernike coeffs, starting at Z_1 to Z_J.
            resid_nm (float) : RMS value of the remainder not fitted by the Zernike.
        """
        nzer = J # number of Zernike modes (from 1 to J included)
        # we suppress the tiptilt from the wavefront
        wavefront_nm = self.phase_generator(self.phase, tiptilt=tiptilt, defoc=defoc) * self.wvl / 2 / np.pi * 1e9
        cubzer = np.zeros((len(wavefront_nm), nzer))
        for k in range(nzer):
            i = k+1 # zernike number
            cubzer[:,k] = zer.zer(self.r, self.theta, i)
        # projection matrix
        projzer = np.linalg.pinv(cubzer)
        # projection
        zervector = projzer.dot(wavefront_nm)
        # computation of RMS value of un-fitted residuals, just for info
        reconstitued_wavefront_nm = cubzer.dot(zervector)
        residues = wavefront_nm - reconstitued_wavefront_nm
        resid_rms = residues.std()
        return zervector, resid_rms


    def estimate_snr(self, conversion_gain_elec_per_adu):
        """
        Estimate the noise of the input images per pixel. The noise is the sum
        of the readout noise of the camera, plus the photon noise. The weight
        is the inverse of the noise. The weight is used to compute the weighted
        least square fit. 

        Args:
            conversion_gain_elec_per_adu (float): conversion gain of the camera
                    in [electrons.adu^-1].
        Returns:
            ndarray: array of weight coefficients, of same shape as the input
                    images. The weight is the inverse of the noise.
        """
        if conversion_gain_elec_per_adu is None:
            # we assume that the maximum valued pixel contains 500,000 electrons
            # (somewhat arbitrary, but difficult to do better !).
            conversion_gain_elec_per_adu = 500.e3 / np.max(self.img)
            line(f"Conversion gain not provided. Using", conversion_gain_elec_per_adu, "e-/ADU")
        # Trick for computing the RON variance: it will be computed only on the
        # non-illuminated pixels, i.e. on the "left part" of the Gaussian
        # distribution of the RON, identified as being the values smaller than
        # the image median value. Moreover this median value is considered as
        # being the average value of the distribution of the RON. Therefore,
        # avg((val[left]-median)**2) is the variance of the RON. 
        valmedian = np.median(self.img, axis=(1,2))
        centred_img = self.img - valmedian[:,None,None] # subtract median of each image
        tmp = centred_img.flatten()
        tmpdark = tmp[tmp < 0] # keep only the non-illuminated pixels (dark pixels)
        varRON = np.mean(tmpdark**2) # variance of the readout noise in adu^2
        line("Estimated RON standard deviation", np.sqrt(varRON), "ADU rms")
        # Computation of the photon noise variance in adu^2 (Poisson law:
        # variance_e- = mean_e-). Note that a^2 = e^2/g^2 = e/g^2 = a.g/g^2 =
        # a/g. This last equation is used below. It seems dimensionally
        # incorrect, but it's not.
        varPHN = np.clip(centred_img, 0, None) / conversion_gain_elec_per_adu
        vartot = varRON + varPHN # total variance in adu^2
        vartot = np.clip(vartot, 1e-10, None) # avoid division by zero
        # The weight is the inverse of the variance, in adu^2. The weight has
        # units of adu^-2, in order that the error of the fit chi2 =
        # (1/nfree).sum(weight*(data - model)**2) has no unit and is close to
        # unity.
        weight = 1./vartot
        return weight

    def search_phase(self, defoc_z_flag=False,
                     focscale_flag=False,
                     optax_flag=False,
                     amplitude_flag=True,
                     background_flag=False,
                     phase_flag=True,
                     illum_flag=False,
                     objsize_flag=False,
                     estimate_snr=False,
                     verbose=False,
                     tolerance=1e-5):
        """
        Central function to search for the phase modal coefficients leading to
        the series of defocused PSFs that best match the input data. The
        function can also tune or search for some other coefficients, such as
        the values of the amount of defocus, the global scale of focus
        coefficients, the tip/tilt coefficients related to each individual
        image, the intensity of each PSF and the background level of each PSF.
        The function uses the lmfit (Levenberg-Marquardt) function to search for
        the best coefficients. The arguments of the function are a list of flags
        that indicate which coefficients should be fitted (when True) and which
        of them remain fixed (False).

        Args:
            defoc_z_flag (bool | list, optional): flag or list of flag for defocus.
                A scalar flag is applied to all defocus coefficients but the first one.
                At least one defocus coefficient must be fixed (False) to avoid 
                singularity in the fit. If a list is provided, it must have the same
                length as the defocus coefficients. Defaults to False.
            focscale_flag (bool, optional): flag for adjusting a global scaling factor
                on the defocus coefficients. Defaults to False.
            wvl_flag (bool, optional): flag for the wavelength. Defaults to False.
            optax_flag (bool | list, optional): flag for the position of the optical axis.
                Defaults to False.
            amplitude_flag (bool | list, optional): flag for image intensity. Recommended
                to set to True. Defaults to True.
            background_flag (bool | list, optional): flag for adjusting the average
                level of the background of each defocus image. Defaults to False.
            phase_flag (bool | list, optional): flag or list that applies to each
                modal coefficient of the phase. Defaults to True.
            illum_flag (bool | list, optional): flag or list for zernike terms of the
                illumination coeffs of the pupil. Defaults to False.
            estimate_snr (bool, optional): flag for estimating (or not) the weighting
                coefficients of the fit, based on the signal-to-noise
            objsize_flag (bool, optional): flag for estimating (or not) the diameter
                of the object
            verbose (bool, optional): flag for printing things when running. Defaults
                to False.
            tolerance (float, optional): set the value where lmfit will stop iterating.
                Defaults to 1e-5.
        Returns:
            None: The function modifies the attributes of the Opticsetup class
                in place. The best coefficients are stored in the class attributes.
        
        Developper tips:
        To add a supplementary parameter to the fit, proceed as follows:
        - function compute_psfs() : 
            * decode new param in the function
            * modify the algo to handle the computation related to the new parameter
        - function search_phase() :
            * add flag parameter in the list of args
            * update comments of the docstring about the new arg
            * encode new param
            * update args in manage_fitting_flags()
            * decode after lmfit()
            * print result after decoding
        - function manage_fitting_flags() :
            * update function consequently
        - visualize_images() :
            * encode properly
        - README.dm
            * update user's manual
        """
        # we first tune the value of ``self.amplitude`` to start the fit
        # with a rather fair guess. The complex wavefront E in the pupil has an
        # expression that is roughly:
        # E = amplitude * self.pupillum * exp(1j*phi),
        # and this goes through an inverse/backward ifft2 transform (that is
        # normalised by 1/n^2) before being squared to get the PSF. Parseval's
        # theorem adapted to discrete inverse/backwrad ifft2 states that the SUM
        # of the square of the ifft2 (i.e. the total energy in the image) is
        # equal to the MEAN of the square of E (i.e. the mean energy in the
        # pupil).
        tmp = np.sum(self.img, axis=(1,2)) - np.median(self.img, axis=(1,2))*self.Ncrop**2
        tmp = tmp * self.N**2 / np.sum(self.pupillum**2)
        self.amplitude = np.sqrt(tmp)
        np.set_printoptions(precision=3)
        line('Amplitude 1st guess', self.amplitude, length=20)
        # create the list of coefficients to be fitted
        coeffs = self.encode_coefficients(self.defoc_z, self.focscale,
                                          self.optax_x, self.optax_y, self.amplitude,
                                          self.background, self.phase, self.illum,
                                          self.object_fwhm_pix)
        # ........ create the list of flags for the coefficients (in the exact
        # same order as in the call to encode_coefficients() just before !!)
        fit_flag = self.manage_fitting_flags(defoc_z_flag, focscale_flag,
                                        optax_flag, amplitude_flag,
                                        background_flag, phase_flag, illum_flag,
                                        objsize_flag)
        # estimate the signal-to-noise ratio of the images in order to provide meaningful w=weights
        if estimate_snr==True:
            self.weight = self.estimate_snr(None)
        else:
            # when snr is not estimated, the fit reduces to a least square
            self.weight = None
        # search for the best coefficients using lmfit
        grint('Starting minimisation process:')
        bestcoeffs = lmfit(compute_psfs, self, coeffs, self.img, w=self.weight,
                           fit=fit_flag, tol=tolerance, verbose=verbose)
        # decode the coefficients and set/store them in the initial object.
        (self.defoc_z, self.focscale, self.optax_x, self.optax_y, self.amplitude,
        self.background, self.phase, self.illum, self.object_fwhm_pix) = self.decode_coefficients(bestcoeffs)

        # ................................. Print the values of found coefficients
        grint( 'Best coefficients found:')
        line('  focus scale', f'{self.focscale:5.3f}', length=20)
        line('  defoc * scale', self.defoc_z*self.focscale*1e3, 'mm', length=20)
        line('  wavelength', self.wvl*1e9, 'nm', length=20)
        line('  x-pos opt axis', self.optax_x*self.rad2pix, 'pix', length=20)
        line('  y-pos opt axis', self.optax_y*self.rad2pix, 'pix', length=20)
        line('  amplitude', self.amplitude, length=20)
        line('  background', self.background, length=20)
        line('  illumination', self.illum, length=20)
        line('  object fwhm', self.object_fwhm_pix, f'pixels ({self.object_shape})', length=20)

        # .................................. Do phase statistics
        phi_pupil_nm = self.phase_generator(self.phase) * self.wvl / (2*np.pi) * 1e9
        # Print the rms value of the phase
        rms_value = np.std(phi_pupil_nm)
        wrms_value = np.sqrt(np.sum(phi_pupil_nm**2 * self.pupillum) / np.sum(self.pupillum))
        grint('Phase statistics :')
        print(f'                              (Raw)                   (Weighted)')
        print(f'  RMS phase value      : {rms_value:6.1f} nm rms             {wrms_value:6.1f} nm rms')
        
        # Remove the tip-tilt component from the phase, and get statistics.
        phi_pupil_nm_notilt = self.phase_generator(self.phase, tiptilt=False) * self.wvl / (2*np.pi) * 1e9
        # Print the rms value of the phase, without tip-tilt
        rms_value_notilt = np.std(phi_pupil_nm_notilt)
        wrms_value_notilt = np.sqrt(np.sum(phi_pupil_nm_notilt**2 * self.pupillum) / np.sum(self.pupillum))
        print(f'  RMS val without TT   : {rms_value_notilt:6.1f} nm rms             {wrms_value_notilt:6.1f} nm rms')

        # Remove tilt and defoc components
        phi_pupil_nm_notiltdef = self.phase_generator(self.phase, tiptilt=False, defoc=False) * self.wvl / (2*np.pi) * 1e9
        # Print the rms value of the phase, without tip-tilt nor defoc
        rms_value_notiltdef = np.std(phi_pupil_nm_notiltdef)
        wrms_value_notiltdef = np.sqrt(np.sum(phi_pupil_nm_notiltdef**2 * self.pupillum) / np.sum(self.pupillum))
        print(f'  RMS val wo TT & Def  : {rms_value_notiltdef:6.1f} nm rms             {wrms_value_notiltdef:6.1f} nm rms')

        # Extract from phase the Zernike coefficients of tip, tilt, focus in nm rms
        ttf = self.convert * self.phase[0:3]
        a2_nmrms = ttf[0] * self.wvl / (2*np.pi) * 1e9
        a2_lD = ttf[0] * 4 / (2*np.pi)
        a2_m = ttf[0] * self.rad2dist
        a2_pix = ttf[0] * self.rad2pix
        a3_nmrms = ttf[1] * self.wvl / (2*np.pi) * 1e9
        a3_lD = ttf[1] * 4 / (2*np.pi)
        a3_m = a3_lD * self.rad2dist
        a3_pix = a3_m * self.rad2pix
        a4_nmrms = ttf[2] * self.wvl / (2*np.pi) * 1e9
        a4_m = ttf[2] * self.rad2z
        a4_pix = a4_m / self.fratio / self.pixelSize
        grint('Values of tilt and defocus of the phase:')
        print( '                     nm rms              l/D             pixels             mm')
        print(f'Tip (horizontal) : {a2_nmrms:8.1f}          {a2_lD:8.3f}          {a2_pix:8.3f}          {a2_m*1e3:8.4f}')
        print(f'Tilt (vertical)  : {a3_nmrms:8.1f}          {a3_lD:8.3f}          {a3_pix:8.3f}          {a3_m*1e3:8.4f}')
        print(f'Defocus          : {a4_nmrms:8.1f}              x             {a4_pix:8.3f}          {a4_m*1e3:8.4f}')

        # ................................. Graphics
        # Create a 2x2 grid of subplots
        plt.figure(1)
        plt.clf()
        zone = ((self.N-self.pdiam*1.1)/2, (self.N+self.pdiam*1.1)/2)
        fig, axes = plt.subplots(2, 2, num=1, figsize=(10, 8))  # nbim rows, 3 columns
        # Plot the retrieved phase first
        phase_map = self.mappy(phi_pupil_nm_notilt)
        im = axes[0, 0].imshow(phase_map.T, cmap='gray', origin='lower')
        axes[0, 0].set_title(f'Retrieved phase [nm] / {wrms_value_notilt:6.1f} nm rms')
        axes[0, 0].set_xlim(zone)
        axes[0, 0].set_ylim(zone)
        axes[0, 0].axis('off')
        self.pupilArtist(axes[0,0])
        fig.colorbar(im, ax=axes[0,0])
        # Plot the retrieved phase with no defocus
        phase_map = self.mappy(phi_pupil_nm_notiltdef)
        im = axes[0, 1].imshow(phase_map.T, cmap='gray', origin='lower')
        axes[0, 1].set_title(f'Retrieved phase [nm] / {wrms_value_notiltdef:6.1f} nm rms')
        axes[0, 1].set_xlim(zone)
        axes[0, 1].set_ylim(zone)
        axes[0, 1].axis('off')
        self.pupilArtist(axes[0,1])
        fig.colorbar(im, ax=axes[0,1])
        # Plot the PSF difference in the third column
        axes[1, 0].imshow(self.mappy(self.pupillum).T, cmap='gray', origin='lower')
        axes[1, 0].set_title('Pupil illumination')
        axes[1, 0].set_xlim(zone)
        axes[1, 0].set_ylim(zone)
        axes[1, 0].axis('off')
        self.pupilArtist(axes[1,0])
        fig.colorbar(im, ax=axes[1,0])
        # Adjust layout
        fig.tight_layout()




def compute_psfs(osetup : Opticsetup, coeffs : np.ndarray):
    """
    This is the function (=the model) that computes the PSFs according to the
    model and coefficients, and that is used in the minimization.
    For compatibility with the minimization procedure this function has only
    2 arguments: the model, and a 1D array of coefficients.

    Args:
        osetup (Opticsetup): object that describes the optical setup (pupil,
                             defocus, fratio, etc.)
        coeffs (np.ndarray): 1d array of coefficients to be fitted. The coefficients
                             are concatenated in a single array. The coeffs order is
                             defined when calling the encode_coefficients() function.
    Returns:
        ndarray: 1d array (flattened) of the series of PSF images. 
    """    
    defoc_z, focscale, optax_x, optax_y, amplitude, background, phase, illum, object_fwhm_pix = osetup.decode_coefficients(coeffs)
    # convert defocus coeffs from delta-Z at focal plane, to radians RMS
    defoc_a4 = focscale * defoc_z / osetup.rad2z # in radians RMS
    # compute the various defocus maps (only on useful pixels)
    ph_defoc = osetup.defoc[None,:] * defoc_a4[:,None] # shape (nbim, nphi)
    ph_optax = osetup.tip[None,:] * optax_x[:,None] + osetup.tilt[None,:] * optax_y[:,None] # shape (nbim, nphi)
    tmp_phase = ph_optax + ph_defoc + osetup.phase_generator(phase)
    # compute the complex field 
    osetup.pupillum = osetup.compute_illumination(illum)
    tmp = osetup.pupillum[None,:]*np.exp(1j*tmp_phase) # shape (nbim, nphi)
    # transform to a two-dimensional map
    tmp = osetup.mappy(tmp) # shape (nbim, N, N)
    # compute the fourier transform of the object
    tf_object = osetup.compute_tf_object(object_fwhm_pix, osetup.object_shape) # object MTF
    tf_object = tf_object * osetup.tf_pixobj
    # allocate the PSF array
    psfs = np.zeros((osetup.nbim, osetup.N, osetup.N)) # shape (nbim, N, N)
    # Loop over the defocus images: compute the PSF for each defocus, then
    # convolution of the psf with the pixel area and with an object, if any. The
    # fourier transform of the object is already computed in the Opticsetup
    # class (tf_pixobj) The amplitude is taken in the square to guarantee a
    # positive value.
    for i in range(osetup.nbim):
        # Inverse fft is used for the sake of getting the right symmetry of the
        # PSF. Inverse fft ("backward") is normalized (divided) by the number of
        # pixels in the image. 
        localpsf = (amplitude[i] * np.abs(np.fft.ifft2( tmp[i,:,:] )))**2
        psfs[i,:,:] = np.fft.fft2(np.fft.ifft2(localpsf) * tf_object).real + background[i]
    return fcrop(psfs, osetup.Ncrop).flatten()




def visualize_images(p : Opticsetup, alpha=1.0):
    """
    Allows to see a summary, presented by a graphic display organized in 3
    columns, of the series of the input data images, the modelled ones, and
    their differences. 

    Args:
        p (Opticsetup): the optical setup object
        alpha (float, optional): a power exponent, to reinforce the visibility
        of the low-level details. Defaults to 1.0.
    """
    def xsoft(image, alpha):
        # Formats the image to get ready for display with improved contrast.
        im = np.fft.fftshift(image.T)
        return np.sign(im) * np.abs(im)**alpha

    # concatenate all relevant coefficients in a single array, to be ready to
    # make a call to the minimized function
    coeffs = p.encode_coefficients(p.defoc_z, p.focscale, p.optax_x, p.optax_y,
                                   p.amplitude, p.background, p.phase, p.illum,
                                   p.object_fwhm_pix)
    # Call to the central model function (returns all coefficients in a row)
    psfs = compute_psfs(p, coeffs)
    # Restore the shape of modelled data to be able to display it properly
    retrieved_psf = np.reshape(psfs, p.img.shape)
    # Recompute the position of the optical axis in pixels in order to plot it
    cc = p.img.shape[1] / 2 # centre of the image
    k = 4 * p.fratio * (1 * p.wvl / 2 / np.pi) / p.pixelSize # convert fact from a2[rad] to pixels
    optax_x, optax_y = (cc-k*p.optax_x, cc-k*p.optax_y) # optical axis position in pixels wrt img centre
    
    # Display the original and retrieved PSFs
    plt.figure(2)
    plt.clf()
    # Create a nbimx3 grid of subplots
    fig, axes = plt.subplots(p.nbim, 3, num=2, figsize=(10, 8))  # nbim rows, 3 columns
    for i in range(p.nbim):
        # Plot the PSF in the first column
        axes[i, 0].imshow(xsoft(p.img[i]-p.background[i], alpha=alpha), cmap='gray', origin='lower')
        axes[i, 0].set_title(f"Input PSF {i+1}")
        axes[i, 0].axis('off')
        axes[i, 0].scatter(optax_x[i:i+1], optax_y[i:i+1], marker='+', color='r') # show optical axis
        # Plot the other PSF in the second column
        axes[i, 1].imshow(xsoft(retrieved_psf[i]-p.background[i], alpha=alpha), cmap='gray', origin='lower')
        axes[i, 1].set_title(f"Retrieved PSF {i+1}")
        axes[i, 1].axis('off')
        axes[i, 1].scatter(optax_x[i:i+1], optax_y[i:i+1], marker='+', color='r') # show optical axis
        # Plot the PSF difference in the third column
        axes[i, 2].imshow(xsoft(p.img[i]-retrieved_psf[i], alpha=1.0), cmap='gray', origin='lower')
        axes[i, 2].set_title(f"Diff input-retrieved {i+1}")
        axes[i, 2].scatter(optax_x[i:i+1], optax_y[i:i+1], marker='+', color='r') # show optical axis
        axes[i, 2].axis('off')
    # Adjust layout
    fig.tight_layout()



