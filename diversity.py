#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 18:08:47 2025

@author: eric gendron

"""

import numpy as np
import zernike as zer
from lmfit_thiebaut import lmfit

import matplotlib.pyplot as plt
plt.ion() # interactive mode on, no need for plt.show() any more


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
                 spiderAngle, spiderArms_m, spiderOffset_m, illum,
                 wvl, fratio, pixelSize, edgeblur_percent, object_fwhm_pix):
        """
        Creation of the Opticsetup class.

        Args:
            img_collection (ndarray | list): data cube (ndefoc, N, N), or a list
                                of images with a size (N, N) obtained experimentally.
            xc (int | None): x coordinate of the center of the image. If None, the center of the
                                image will be used.
            yc (int | None): y coordinate of the center of the image. If None, the center of the
                                image will be used.
            N (int | None) : size of the square image. If None, the largest square format that fits in the
                                original image will be used. If N is odd, it will be forced to even.
            defoc_z (ndarray | list): list of the amount of defocus in [m] for each image.
            pupilType (int): 0: disk/ellipse, 1: polygon, 2: ELT
            flattening (float): flattening factor of the pupil. Applies to circular (disk) apertures
                                as well as to polygonal apertures. The flattening operates in a direction
                                perpendicular to the angle of the pupil. The flattening is defined as
                                the ratio of the minor axis to the major axis of the ellipse.
            obscuration (float): obscuration factor of the pupil. Applies to circular (disk) apertures
                                as well as to polygonal apertures.
            angle (float): angle of the pupil in radians. 
            nedges (int): number of edges of the polygon pupil. Only useful for polygon pupil.
            spiderAngle (float): angle of the spider in radians, counted from the axis defined by the
                                angle of the pupil. 
            spiderArms_m (list): list of the widths of each spider arms in [m]. The number of arms is
                                equal to the number of elements in this list. An empty list [] means
                                no spider. The arms are assumed to be straight lines.
            spiderOffset_m (list): list of the offsets the line of each spider arm wrt the centre of the
                                pupil, in [m]. The length of the list must be equal to the length of the
                                spiderArms_m list. 
            illum (list | ndarray): illumination coefficients of the pupil. The illumination map is
                                described using Zernike coefficients. The first coefficient is the
                                piston term, Z_1(r,t) = 1.00, and correspond to a flat illumination.
                                Therefore for a simple, flat illumination, the list should just be [1.0].
                                The list can be of any arbitrary length.
            wvl (float): wavelength of the light in [m]. 
            fratio (float): focal ratio of the setup forming the images of the data cube.
            pixelSize (float): size of the pixel in [m].
            edgeblur_percent (float): percentage of the edge blur applied to the edges of the pupil.
            object_fwhm_pix (float): FWHM of the object in [pixels]. The object is assumed to be either a Gaussian
                                or a disk. A value of 0.0 means an infinitely small object.
        """        
        self.img = check_image_format(img_collection, xc, yc, N) # list of images
        self.nbim, self.N, _ = self.img.shape # number of images

        if len(defoc_z)==self.nbim:
            self.defoc_z = np.array(defoc_z)
        else:
            raise ValueError('Number of images ({self.nbim}) does not match number of defocus ({len(defoc_z)}).')
        
        self.pupilType = ['disk','polygon','ELT'][pupilType]
        self.flattening = flattening
        self.obscuration = obscuration
        self.angle = angle
        self.nedges = nedges # only useful for polygon pupil
        
        self.spiderAngle = spiderAngle
        self.spiderArms_m = spiderArms_m
        self.spiderOffset_m = spiderOffset_m
        self.nspider = len(self.spiderArms_m)
        self.edgeblur = np.maximum(edgeblur_percent, 1e-6) # in percent

        self.illum = illum # zernike list of illum, starting with piston

        self.wvl = wvl # wavelength in [m]

        self.fratio = fratio # ratio f/D
        self.pixelSize = pixelSize # size of pixels in [m]

        # Evaluate the number of pixels required in the pupil diameter to
        # match the plate scale of the images
        self.pdiam = self.compute_pupil_diam()
        print(f"Number of pixels oh phase in the pupil diameter (f/{self.fratio:6.3f}): {self.pdiam:7.3f}")
        print(f"Angular size of a pixel of phase in the pupil plane: f/{self.fratio*self.pdiam:7.3f}")

        # standard variables
        # create variables x, y over whole square support
        tmp = np.arange(self.N) - self.N/2.0
        self.x, self.y = np.meshgrid(tmp, tmp, indexing='ij')

        # pupil model
        self.pupilmap = self.pupilModel()
        self.idx = self.pupilmap.nonzero() # index of pixels of the useful pupil part

        # standard variables defined only over the pupil area for zernike evaluation
        self.tip = self.x[self.idx] / (self.pdiam/4)
        self.tilt = self.y[self.idx] / (self.pdiam/4)
        self.r = np.sqrt(self.x[self.idx]**2 + self.y[self.idx]**2) / (self.pdiam/2)
        self.theta = np.arctan2(self.y[self.idx], self.x[self.idx])
        self.defoc = zer.zer(self.r, self.theta, 4)

        # number of phase points that will be treated.
        # Here we are going to define a new basis for the phase, that will be obtained
        # by diagonalisation of the matrix of the pairwise distances**(5/3) between the useful
        # pixels of the pupil.
        nphi = self.idx[0].size
        print(f"Number of phase points in the pupil: {nphi}")
        mat = ((self.tip[:,None]-self.tip[None,:])**2 + (self.tilt[:,None]-self.tilt[None,:])**2)**(5/6)
        P = np.eye(nphi) - np.ones((nphi,nphi))/nphi # piston-removal matrix
        mat = P @ mat @ P
        # diagonalise the matrix. The eigenvalues are sorted in increasing order
        s_eig, self.phase_basis = np.linalg.eigh(mat)
        self.phase = np.zeros(10)

        # calculate the pupil illumination using the Zernike polynomials just over the pupil area
        self.pupillum = self.compute_illumination(self.illum)

        self.wrap = []

        # define some useful variables that will be used later in the lmfit
        self.a2tip = np.zeros(self.nbim)
        self.a2tilt = np.zeros(self.nbim)
        self.amplitude = np.ones(self.nbim)
        self.background = np.zeros(self.nbim)

        # compute the Fourier transform of the square shape covered by a single pixel
        # multiplied by the Fourier transform of an object with a given FWHM. When the
        # FWHM is zero the function compute_tf_object returns the scalar 1.0, to fasten
        # the computation.
        self.tf_pixobj = self.compute_tf_object(1.0, 'square') # pixel MTF
        self.tf_pixobj = self.tf_pixobj * self.compute_tf_object(object_fwhm_pix, 'gaussian') # object MTF




    def compute_tf_object(self, object_fwhm_pix, type='gaussian'):
        """
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


    def phase_generator(self, vector):
        """
        Convert the vector of modal coefficients to a zonal (=pixel list) representation of the phase.
        The phase is computed using the phase basis, which is a set of orthogonal modes
        defined over the pupil area. The phase is computed only over the pixels of the pupil area.

        Args:
            vector (ndarray)  : vector of modal coefficients to be converted to a zonal (=pixel list)
                                representation of the phase.
        Returns:
            ndarray: list of the modal phase values in the pupil
        """        
        n = np.minimum(vector.size, self.phase_basis[0,:].size)
        # convert the vector to a phase map
        phase = self.phase_basis[:,:n] @ vector[:n]
        return phase


    def compute_illumination(self, zernik_illum_array):
        """
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
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        u = self.x*c + self.y*s
        v = self.x*(-s/self.flattening) + self.y*(c/self.flattening)
        blur = self.edgeblur/100*self.pdiam
        # width and middle of donut
        Am = self.pdiam*(1 + self.obscuration)/4 # middle between R and R.robs
        Dm = self.pdiam*(1 - self.obscuration)/2 # width of the donut
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
            reliefSpiderLeg = (np.abs( u * ss - v * cc + self.spiderOffset_m[i]*self.pdiam) - self.spiderArms_m[i]*self.pdiam/2.0) / blur
            no_spider_zone = (u * cc + v * ss) < 0  # identify the right pupil half where the spider arm is
            reliefSpiderLeg[no_spider_zone] = 42. # set to any number greater than 1.0
            relief = np.minimum(relief, reliefSpiderLeg)

        img = np.clip(relief + 0.5, 0, 1)
        return img
    

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
        defoc_a4 = self.defoc_z / (16 * np.sqrt(3) * self.fratio**2) * (2*np.pi) / self.wvl # in radians RMS
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
        output = ()
        for deckey in self.wrap:
            datatype, length, shape = deckey
            if datatype=='scalar':
                output = output + (itemlist[ptr],)
            elif datatype=='list':
                output = output + (list(itemlist[ptr:ptr+length]),)
            elif datatype=='ndarray':
                tmp = np.reshape(itemlist[ptr:ptr+length], shape)
                output = output + (tmp,)
            elif datatype=='1darray':
                output = output + (itemlist[ptr:ptr+length],)
            else:
                raise ValueError(f'Decoding keyword {datatype} is unknown. Cant decode the sequence.')
            ptr = ptr+length
        return output


    def zerphase(self, J=21):
        """
        Compute the decomposition of the phase into Zernike coefficients.

        Warning: This decomposition only makes sense when the pupil is circular.
        Results obtained for other pupil shapes may sometimes appear to make
        sense, but must be treated with extreme caution. Zernike modes are
        defined as usual as Zi(r/R, theta), with theta=0 directed along the
        X-axis (horizontally, rather than along the main axis of the pupil). The
        unitary disk where the Zernike modes are defined, is mapped onto the
        circle of a diameter defined by the f-ratio. Then, the real pupil of the
        system acts as a mask over this. The decomposition is performed by
        de-coupling from the non-orthogonality of the polynomials on the
        support/mask they are computed and using the native pupil sampling, in
        such a way that their weighted sum restitutes back the retrieved phase
        over the system pupil. Values outside the system pupil are meaningless
        (they should not be computed). On non-circular pupils, RMS values of the
        coefficients are meaningless.
        As only a limited number of polynomials are involved (21 by default),
        their combination may not fully recreate the initial wavefront.
        Therefore the routine also returns the RMS value of the amount of
        leftovers that were ignored in the expansion, for information.

        Args:
            J (int): Optional. Number of the last Zernike mode that is
                     considered in the decomposition. Default is 21.
        Return:
            zervector (1D ndarray) : array of Zernike coeffs, starting at Z_2 to Z_J.
            resid_nm (float) : RMS value of the remainder not fitted by the Zernike.
        """
        nzer = J-1 # number of Zernike modes (from 2 to J included)
        wavefront_nm = self.phase_generator(self.phase) * self.wvl / 2 / np.pi * 1e9
        cubzer = np.zeros((len(wavefront_nm), nzer))
        for k in range(nzer):
            i = k+2 # zernike number
            cubzer[:,k] = zer.zer(self.r, self.theta, i)
        # projection matrix
        projzer = np.linalg.pinv(cubzer)
        # projection
        zervector = projzer.dot(wavefront_nm)
        # computation of RMS value of un-fitted residuals, just for info
        residues = cubzer.dot(zervector) - wavefront_nm
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
            print(f"Conversion gain not provided. Using {conversion_gain_elec_per_adu:.2e} e-/ADU.")
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
        print(f"RON standard deviation: {np.sqrt(varRON):.2e} adu rms")
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
                     fratio_flag=False,
                     wvl_flag=False,
                     tiptilt_flag=False,
                     amplitude_flag=True,
                     background_flag=False,
                     phase_flag=True,
                     illum_flag=False,
                     estimate_snr=False,
                     verbose=False,
                     tolerance=1e-5):
        """
        Central function to search for the phase modal coefficients leading to
        the series of defocused PSFs that best match the input data. The
        function can also search for some other coefficients, such as the values
        of the amount of defocus, the focal ratio, the wavelength, the tip/tilt
        coefficients related to each individual image, the intensity of each PSF
        and the background level of each PSF. The function uses the lmfit
        (Levenberg-Marquardt) function to search for the best coefficients.
        The arguments of the function are a list of flags that indicate which
        coefficients should be fitted (when True) and which of them remain fixed
        (False).

        Args:
            defoc_z_flag (bool | list, optional): flag or list of flag for defocus.
                A scalar flag is applied to all defocus coefficients but the first one.
                At least one defocus coefficient must be fixed (False) to avoid 
                singularity in the fit. If a list is provided, it must have the same
                length as the defocus coefficients. Defaults to False.
            fratio_flag (bool, optional): flag for f-ratio. Defaults to False.
            wvl_flag (bool, optional): flag for the wavelength. Defaults to False.
            tiptilt_flag (bool | list, optional): flag for tip and tilt. Defaults to False.
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
            verbose (bool, optional): flag for printing things when running. Defaults
                to False.
            tolerance (float, optional): set the value where lmfit will stop iterating.
                Defaults to 1e-5.
        Returns:
            None: The function modifies the attributes of the Opticsetup class
                in place. The best coefficients are stored in the class attributes.
        """
        # we first tune the value of ``self.amplitude`` to start the fit
        # with a rather fair guess. The complex wavefront E in the pupil has an
        # expression that is roughly:
        # E = amplitude * self.pupillum * exp(1j*phi),
        # and this goes through an inverse/backward ifft2 transform (that
        # is normalised by 1/n^2) before being squared to get the PSF.
        # Parseval's theorem adapted to discrete inverse/backwrad ifft2 states
        # that the SUM of the square of the ifft2 (i.e. the total energy in the
        # image) is equal to the MEAN of the square of E (i.e. the mean energy
        # in the pupil).
        tmp = np.sum(self.img, axis=(1,2)) - np.median(self.img, axis=(1,2))*self.N**2
        tmp = tmp * self.N**2 / np.sum(self.pupillum**2)
        self.amplitude = np.sqrt(tmp)
        print(f'Initial amplitude guess: {self.amplitude}')
        # create the list of coefficients to be fitted
        coeffs = self.encode_coefficients(self.defoc_z, self.fratio, self.wvl,
                                          self.a2tip, self.a2tilt, self.amplitude,
                                          self.background, self.phase, self.illum)
        # ........ create the list of flags for the coefficients
        list_flag = np.full(len(self.defoc_z), defoc_z_flag)
        # exception for defocus: one of the coefficients must be fixed
        if np.sum(list_flag)==self.nbim:
            list_flag[0] = False
        list_flag = np.concatenate((list_flag, np.full(1, fratio_flag)))
        list_flag = np.concatenate((list_flag, np.full(1, wvl_flag)))
        # tip-tilt case
        tiptilt_list_flag = np.full(len(self.a2tip), tiptilt_flag)
        # same exception as for defocus: one of the tiptilt coefficients must be fixed
        if np.sum(tiptilt_list_flag)==self.nbim:
            tiptilt_list_flag[0] = False
        list_flag = np.concatenate((list_flag, tiptilt_list_flag)) # this is for tip
        list_flag = np.concatenate((list_flag, tiptilt_list_flag)) # same as tip: this is for tilt
        # amplitude case
        list_flag = np.concatenate((list_flag, np.full(len(self.amplitude), amplitude_flag)))
        list_flag = np.concatenate((list_flag, np.full(len(self.background), background_flag)))
        list_flag = np.concatenate((list_flag, np.full(len(self.phase), phase_flag)))
        # illumination case. First coeff (=piston=flat illum) is never fitted.
        illum_list_flag = np.full(len(self.illum), illum_flag)
        illum_list_flag[0] = False
        list_flag = np.concatenate((list_flag, illum_list_flag))
        # get the indices of the coefficients to be fitted
        (fit_flag,) = list_flag.nonzero()
        # estimate the signal-to-noise ratio of the images in order to provide meaningful w=weights
        if estimate_snr==True:
            weight = self.estimate_snr(None)
            self.weight = weight
            # plt.figure(1)
            # plt.clf()
            # plt.imshow(weight[0].T, origin='lower', cmap='gray')
            # plt.title('Weight image')
            # plt.colorbar()
            # plt.pause(3)
            # input('Press Enter to continue...')
        else:
            # when snr is not estimated, the fit reduces to a least square
            weight = None
        # search for the best coefficients using lmfit
        bestcoeffs = lmfit(compute_psfs, self, coeffs, self.img, w=weight, fit=fit_flag, tol=tolerance, verbose=verbose)
        # decode the coefficients
        self.defoc_z, self.fratio, self.wvl, self.a2tip, self.a2tilt, self.amplitude, self.background, self.phase, self.illum = self.decode_coefficients(bestcoeffs)
        # Print the values of the best coefficients
        print(f'Best coefficients found:')
        print(f'  defocus        : {self.defoc_z}')
        print(f'  fratio         : {self.fratio:5.2f}')
        print(f'  wvl            : {self.wvl*1e9} nm')
        print(f'  tip            : {self.a2tip}')
        print(f'  tilt           : {self.a2tilt}')
        print(f'  amplitude      : {self.amplitude}')
        print(f'  background     : {self.background}')
        print(f'  illumination   : {self.illum}')
        # Display an image of the retrieved phase, converted to a 2D pupil map
        # and scaled to nanometers
        phi_pupil_nm = self.phase_generator(self.phase) * self.wvl / (2*np.pi) * 1e9
        # Print the rms value of the phase
        rms_value = np.std(phi_pupil_nm)
        print(f'  RMS phase value (raw): {rms_value:5.1f} nm rms')
        rms_value = np.sqrt(np.sum(phi_pupil_nm**2 * self.pupillum) / np.sum(self.pupillum))
        print(f'  RMS phase value (weighted): {rms_value:5.1f} nm rms')
        
        # Remove the tip-tilt component from the phase, and get statistics.
        phi_pupil_nm_tt = self.tip * np.sum(phi_pupil_nm*self.tip)/np.sum(self.tip**2) + self.tilt * np.sum(phi_pupil_nm*self.tilt)/np.sum(self.tilt**2)
        phi_pupil_nm_notilt = phi_pupil_nm - phi_pupil_nm_tt
        phi_pupil_nm_notilt = phi_pupil_nm_notilt - np.mean(phi_pupil_nm_notilt)
        # Print the rms value of the phase, without tip-tilt
        rms_value = np.std(phi_pupil_nm_notilt)
        print(f'  RMS phase value without TT (raw): {rms_value:5.1f} nm rms')
        rms_value = np.sqrt(np.sum(phi_pupil_nm_notilt**2 * self.pupillum) / np.sum(self.pupillum))
        print(f'  RMS phase value without TT (weighted): {rms_value:5.1f} nm rms')

        # Graphics
        phase_map = self.mappy(phi_pupil_nm_notilt)
        plt.clf()
        plt.imshow(phase_map.T, origin='lower', cmap='gray')
        plt.title('Retrieved phase map [nm]')
        plt.colorbar()




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
    defoc_z, fratio, wvl, a2tip, a3tilt, amplitude, background, phase, illum = osetup.decode_coefficients(coeffs)
    # convert defocus coeffs from delta-Z at focal plane, to radians RMS
    defoc_a4 = defoc_z / (16 * np.sqrt(3) * fratio**2) * (2*np.pi) / wvl # in radians RMS
    # compute the various defocus maps (only on useful pixels)
    ph_defoc = osetup.defoc[None,:] * defoc_a4[:,None] # shape (nbim, nphi)
    ph_tiptilt = osetup.tip[None,:] * a2tip[:,None] + osetup.tilt[None,:] * a3tilt[:,None] # shape (nbim, nphi)
    tmp_phase = ph_tiptilt + ph_defoc + osetup.phase_generator(phase)
    # compute the complex field 
    osetup.pupillum = osetup.compute_illumination(illum)
    tmp = osetup.pupillum[None,:]*np.exp(1j*tmp_phase) # shape (nbim, nphi)
    # transform to a two-dimensional map
    tmp = osetup.mappy(tmp) # shape (nbim, N, N)
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
        psfs[i,:,:] = np.fft.fft2(np.fft.ifft2(localpsf) * osetup.tf_pixobj).real + background[i]
    return psfs.flatten()




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

    coeffs = p.encode_coefficients(p.defoc_z, p.fratio, p.wvl,
                                    p.a2tip, p.a2tilt, p.amplitude,
                                    p.background, p.phase, p.illum)
    psfs = compute_psfs(p, coeffs)
    retrieved_psf = np.reshape(psfs, p.img.shape)
    # Display the original and retrieved PSFs
    plt.figure(2)
    plt.clf()
    # Create a nbimx3 grid of subplots
    fig, axes = plt.subplots(p.nbim, 3, num=2, figsize=(10, 8))  # nbim rows, 3 columns
    for i in range(p.nbim):
        # Plot the PSF in the first column
        axes[i, 0].imshow(xsoft(p.img[i], alpha=alpha), cmap='gray', origin='lower')
        axes[i, 0].set_title(f"Input PSF {i+1}")
        axes[i, 0].axis('off')
        # Plot the other PSF in the second column
        axes[i, 1].imshow(xsoft(retrieved_psf[i], alpha=alpha), cmap='gray', origin='lower')
        axes[i, 1].set_title(f"Retrieved PSF {i+1}")
        axes[i, 1].axis('off')
        # Plot the PSF difference in the third column
        axes[i, 2].imshow(xsoft(p.img[i]-retrieved_psf[i], alpha=1.0), cmap='gray', origin='lower')
        axes[i, 2].set_title(f"Difference PSF {i+1}")
        axes[i, 2].axis('off')
    # Adjust layout
    fig.tight_layout()
