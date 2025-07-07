# Hello, user.

This module performs phase retrieval from a series of out-of-focus focal plane
images.

The module contains the Opticsetup class, which has methods to simulate and to
fit a series of defocused images (at least two). The Opticsetup class is used to
create a model of the optical setup, including in particular the pupil function,
the defocus coefficients, the illumination coefficients, and various other
details that define the whole model of the image formation process.

The phase retrieval consist in iterating on the phase coefficients of this model
until the produced images become as close as possible to the user's data, in a
(weighted) least-square sense. The minimisation is performed using a Levenberg-
Marquardt algo.


# User manual

The main code is in `diversity.py`, the rest is utilities and libraries.
The `test` folder contains rubish things, do not use it.
Just start with a
```python
import diversity as div
```
then create your optical setup (see paragraph below) and then search the phase
(paragraph after).


## Definition of the optical setiup

The optical setup is an object that contains all the information required to
form/compute an image similar to that of the input data, but it contains also
the experimental images to be processed.

1) The user must provide a data cube or a list of defocused images to be fitted,
   with the defocus in the first dimension, and the image size in the second and
   third dimensions. The images must be background subtracted, flat-fielded when
   needed, averaged, with dead pixels removed.
   The images may possibly be rectangular, but they will be cropped by the
   procedure to the largest possible square format (see below).

2) Optionally the user may also provide the coordinates of the peak of the
   on-focus image. If not provided, the centre of the image will be assumed.
   A square-format image will be cropped/extracted, centred on that position.

3) Optionally the size N of the desired square image can be also set.
   - If not set (N=None, by default), the largest possible square format that
     fits into the original image will be used. The data and the computation
     supports (FFTs) will all be NxN.
   - If N is set and if the NxN format fits into the data image, then it will be
     used. The data and the computation supports (FFTs) will all be NxN as the
     case before.
   - If N is set but the N × N format exceeds the size of the data image, the
     latter will be cropped to a suitable smaller size. However, the computation
     support size for Fourier transforms will still be NxN. This feature enables
     work with truncated data images without any truncation effects in the
     numerical modelling (only the common part is compared to the data at the
     end).
   When possible, choosing a value N=2^n is an advantage for speeding up the
   FFTs.
   The images shall not be too large, they shall be restricted to an area where
   some relevant information appears about the PSF, without cutting the
   interesting features of the PSF too much. If the relevant information (above
   1 sigma of noise) is comprised in an area of diameter D, an image size of the
   order of 1.5 to 2 D is recommended.
   The user may either format/crop his own images beforehand and provide the
   function with appropriate images size, or let the function do it.


   Examples of use:
   ```python
   # Let's assume 3 images, the psf peak is centered on (32,32)
   # The image size is img_collection.shape = (3, 64, 64)
   # The image size 64x64 is suitable to work with.
   mysetup = div.Opticsetup(img_collection, xc=None, yc=None, N=None, ...
   ```

   ```python
   # Let's assume 3 images, psf peak is centered, i.e. on (240, 320)
   # The image size is img_collection.shape = (3, 480, 640)
   # Here, it's advisable to crop the image to a smaller format, e.g. 64x64 
   mysetup = div.Opticsetup(img_collection, xc=None, yc=None, N=64, ...
   ```

   ```python
   # Let's assume 3 images, with a psf peak centered at (135,450)
   # The image size is img_collection.shape = (3, 480, 640)
   mysetup = div.Opticsetup(img_collection, xc=135, yc=450, N=64, ...
   ```

   ```python
   # Let's assume 3 images, with a psf peak centered at (20,19)
   # The image size is narrow and closely brackets the
   # the PSF image, img_collection.shape = (3, 40, 40)
   # Then it's advisable to impose a larger format for the computation
   mysetup = div.Opticsetup(img_collection, xc=None, yc=None, N=64, ...
   ```

4) The user must provide the defocus amplitudes, in the same order as the images
   in the data cube. The coefficients are the defocus distance in [m] at the
   level of the focal plane where the detector is.
   Assuming that the light propagates from the exit pupil to the exit focal plane,
   - positive value of the focus is for an observation plane placed downstream
     the nominal exit focal plane,
   - negative value for an upstream focal plane.

   Example:
   ```python
   # Examples with defocus of 0, -0.5mm and 1.0mm
   # Mind the units !
   mysetup = div.Opticsetup(img_collection, xc=None, yc=None, N=None, defoc_z=[0.0, -0.5e-3, 1.0e-3], ...)
   ```

5) The user must provide the pupil type. The pupil type is defined by an integer
   as follows:
   - 0: disk/ellipse
   - 1: regular polygon
   - 2: ELT shape (not implemented yet)

6) The user must provide the flattening factor of the pupil, useful for defining
   elliptical pupils. This factor will impact everything that is contained in
   the pupil, whether it is circular or polygonal. Spiders and obscuration are
   affected as well. The flattening operates in a direction perpendicular to a
   "main pupil axis", which orientation is defined the angle of the pupil (see
   6. below). The flattening factor can possibly be greater than 1.0 (therefore
   acting as an expansion factor rather than flattening).

7) The user must provide the central obscuration diameter of the pupil. This
   factor will create a central obscuration with the same shape as the pupil. It
   also  applies  to polygonal  pupils.  It  has no unit,  it is  expressed as a
   fraction of the pupil diameter. 

8) The user must provide the angle of the "main pupil axis" in radians, positive
   angles rotate the whole pupil (shape, spiders, obscuration) counter
   clockwise. An angle of 0.0 rad means that the "main pupil axis" is aligned
   with the X axis (horizontal).

9) The user must provide the number of edges of the polygon pupil. This
   parameter is only useful for polygon pupils (pupil type 1), it is ignored
   otherwise. The number of edges must be > 2. The polygon is a regular one
   (vertices equally spaced). The flattening factor can be used to distort it.

10) The user must provide the spider angle in radians. The spider angle affects
   the clocking of all the spider arms relative to the "main pupil axis", i.e.
   to the rest of the pupil.

11) The user must provide the width of each spider arms, expressed as a fraction
   of the pupil diameter measured along the "pupil main axis" (i.e. the
   non-flattened axis). The spider arms are straight lines. The number of arms
   is equal to the number of elements in this list. An empty list [] means no
   spider.

12) The user must provide the offsets of the line of each spider arm wrt the
    center of the pupil, expressed as a fraction of the pupil diameter measured
    along the "pupil main axis" (i.e. the non-flattened axis). The length of the
    list must be equal to the length of the spiderArms list. Use [] if there is
    no spider. Use [0,0,..,0] when everything is centred.

13) The user must provide the illumination coefficients of the pupil. The
    illumination map is described using Zernike coefficients (exceptionally they
    are not meant to describe a phase here!). The first coefficient is the
    piston term, Z_1(r,t) = 1.00, and therefore correspond to a flat
    illumination. Therefore for a simple flat illumination, the list should just
    be [1.0]. The list can be of any arbitrary length. The order of Zernike
    modes is that of (Noll 1976), i.e. piston, tip, tilt, defocus, 2 astigs,
    etc. An edge darkening (gaussian illumination?) can be modelled by
    [1,0,0,-0.1] for instance. This parameter is part of the parameters that can
    be fitted later on. The length of the array will indicate how many Zernike
    coefficients will be searched for.

14) The user must provide the wavelength of the light in [m].

15) The user must provide the focal ratio of the setup forming the images of the
    data cube. The product with the wavelength gives the diameter of the
    diffraction pattern in meters. This focal ratio is the ratio f/D, where D is
    the diameter of the pupil along the direction of the pupil axis (defined by
    the pupil angle), i.e. in the direction that is not affected by the
    flattening coefficient.

16) The user must provide the size of the pixel in [m].

17) The user must provide the percentage of edge blur in [%] applied to the
    edges of the pupil. This parameter is not physical, it is used to limit the
    numerical impact of diffraction effects at very high spatial frequencies and
    avoid aliasing and numerical image wrapping due to FFT in the focal plane. A
    value of 3 to 5.0 is usually adequate.

18) The user must provide the FWHM of the object in [pixels]. The object is
    assumed to be gaussian. Floating-point values smaller than 1 pixel are
    possible without any sampling issues because the computation is done
    directly in the Fourier space (multiplication by a broad gaussian function).
    A value of 0.0 means an infinitely small object (i.e. point-source). The
    code is prepared for circular and square objects, but the feature is not yet
    implemented. Note that the same code is used internally to simulate the
    convolution by the pixel area function, using a square object with a FWHM of
    1.0 pixel. Therefore, this "object fwhm feature" can also be used to
    simulate the impact of detector pixels with an influence shape that would be
    larger than 1 pixel. 

An example of use is given below, with a fairly elliptic pupil barred with 2
spiders:

```python
import diversity as div
img_collection = ...(user code here)... 
mysetup = div.Opticsetup(img_collection, xc=None, yc=None, N=None, # image data can be used "as is"
                        defoc_z=[0.0, -0.5e-3, 1.0e-3], # 3 images were given
                        pupilType=0,                    # pupil type: disk/ellipse
                        flattening=2/3.0,               # ellipse
                        obscuration=0.12,               # with an obscuration of 12%
                        angle=0.0,                      # major axis aligned with X axis
                        nedges=0,                       # ignored
                        spiderAngle=0.0,                # spider arms aligned with X axis
                        spiderArms=[0.035, 0.035],    # 2 arms of 3.5% of pupil diam ..
                        spiderOffset=[0.0, 0.0],      # .. pointing toward the centre
                        illum=[1.0],                    # flat illumination
                        wvl=550e-9,                     # wavelength of 550nm
                        fratio=18.0,                    # f/D = 18
                        pixelSize=7.4e-6,               # pixel size of 7.4um
                        edgeblur_percent=3.0,           # 3% edge blur
                        object_fwhm_pix=0.4)            # object is 0.4 pixel wide
```

After doing this, we recommend the user to check that the images are properly centered
and cropped by displaying the images `mysetup.img[k]`.
WARNING: Don't be surprised, the image centre is expected to be spread at the 4 image
corners (for reasons related to Fourier transformation).
```python
plt.imshow( mysetup.img[0].T, origin='lower', cmap='gray')
```

Second, it is advisable to look at the image of the pupil, to check that the
parameters are correct. The pupil image is available in the class:
```python
plt.imshow( mysetup.pupilmap.T, origin='lower', cmap='gray')
```

The pupil with the illumination can be displayed using the following command: 
```python
plt.imshow( mysetup.mappy(mysetup.pupillum).T, origin='lower', cmap='gray')
```


## Searching for the aberration coefficients:

The search_phase() function is the central function to perform phase diversity
fitting. It is invoked as follows:
```python
   mysetup.search_phase(defoc_z_flag=False,
                           fratio_flag=False,
                           wvl_flag=False,
                           tiptilt_flag=True,
                           amplitude_flag=True,
                           background_flag=False,
                           phase_flag=True,
                           illum_flag=False,
                           estimate_snr=False,
                           verbose=True,
                           tolerance=1e-5)
```
In the above command the user can select which parameters are to be fitted.
Once executed, the function modifies the attributes of the object, which can be
accessed using the following commands:
```python
   mysetup.phase # modal coefficients of the phase
   mysetup.phase_generator(mysetup.phase) # zonal representation of the phase, all phase points in a row
   mysetup.mappy(mysetup.phase_generator(mysetup.phase)) # human-readable/plottable 2d representation of the phase
```

The function `mysetup.search_phase()` starts from a first initial guess of the
phase. This guess is only made of zeros and has 10 modal terms. It is possible
to provide a different initial guess by setting the attribute mysetup.phase to a
different value before invoking `mysetup.search_phase()`.
```python
   mysetup.phase = np.zeros(20) # will search 20 modal coefficients instead of 10.
```

A verification function allows the user to estimate the validity of the result.
The function below allows the user to see a summary, presented by a graphic
display organized in 3 columns, of
- the series of the input data images, 
- the modelled ones, 
- and their differences.
```python
   div.visualize_images(mysetup, alpha=0.5) # square-root scaling of gray levels
```
The parameter `alpha` is a power exponent on the images, that enhances the
visibility of the low-level details. Defaults to 1.0.



## Conventions for image plotting

All these routines have been written using the [x,y] convention. The first
(left) index is the horizontal axis X positive to the right, the second (=right,
y) is the vertical axis Y positive to the top. Therefore, all the images must be
displayed with the `plt.imshow(...T, origin='lower')` directive, using
imperatively a transpose and the `origin='lower'` option.
Consider not using anything else but this (below) in the future for plotting images:
```python
def pli(im, *args, **kwargs):
    """
    This function allows the user to write a clean code where any reference to
    a coefficient [i,j] of an image appears at (i,j) coordinates on the matplotlib
    graphic window, and superimposes naturally and logically with the axes of any
    forthcoming plt.plot or plt.scatter or else. Prevent generations of PhD
    students and postdocs from nervous crisis and brain hyperheating after randomly
    and desperately filling the code with axis swaps, .T and ::-1.
    Warning: Use for images ONLY, not for matrices.
    """
    plt.imshow(im.T, origin="lower", *args, **kwargs)
```
