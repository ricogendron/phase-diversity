

msg_shannon_violated = """
According to the input arguments, the diffraction pattern is sampled using only
%.1f pixels. This violates Nyquist/Shannon theorem, the image is
undersampled. Therefore, the results are not guaranteed. Could there be a
problem with the input parameters ? You should either increase the wavelength
and the F-ratio, or decrease the pixel size.
"""

msg_low_ndof = """
There are only %d degrees of freedom available for the phase in the pupil.
This is actually a pretty small number. Is this normal, are you sure this is
what you really want ? This situation may occur for several reasons, but
fundamentally, it is because the area covered by the field of view is less than
%d times the size of the system's diffraction pattern.
Perhaps this is what you actually want.
If not, you may wish to consider changing or verifying your input parameters.
This situation probably occurs because one (or more) of the following applies:
 - Your input images are too small and should be cropped to a more reasonable
   size (a larger FoV is required). Has the parameter N=... been set
   appropriately ? Are the input images correct ?
 - The wavelength is incorrect (mind the units : meters).
 - The f-ratio is incorrect (too large)
 - The pixel size is incorrect (too small)
 - Your images are severely oversampled: what about binning your pixels ?
"""


# Case "eigenfull" with more than 1500 points
msg_high_ndof = """
You have %d degrees of freedom for the phase in the pupil, and you have chosen
to exploit all of them by selecting to use a full modal basis. This is actually
quite a lot. The computation at initialisation may be quite intensive, it may
take %.1f minutes. Are you sure this is what you really want ?
This situation may occur for several reasons, but fundamentally, it is because
the area covered by the field of view is at least %d times the size of the
system's diffraction pattern.
Perhaps this is what you actually want.
If not, you may wish to consider changing the input parameters. This situation
probably occurs because one (or more) of the following applies:
 - Your input images are very large and should be cropped to a more
   reasonable size (a large fraction of the FoV is useless). Simply use the
   parameter N=.. to crop the images appropriately.
 - The parameter N=... has been set to a value that is too large.
 - The wavelength is incorrect (too short, resulting in a small Airy pattern).
 - The f-ratio is incorrect (too small)
 - The pixel size is incorrect (too large, mind the units : meters)
 - Your images are severely undersampled: then we recommend avoiding any phase
   diversity attempt on such data (at the risk of great frustration).
"""



msg_huge_ndof = """
You have %d degrees of freedom for the phase in the pupil. This is actually
quite a lot. The computation may be quite intensive. Are you sure this is what
you really want ? It will occupy %d MBytes of RAM. This situation may
occur for several reasons, but fundamentally, it is because the large area
covered by the field of view is at least %d times the size of the system's
diffraction pattern.
Perhaps this is actually what you want : in this case, you should perhaps use
the argument basis='zernike' rather than 'eigen'.
If it is not, then you may wish to consider changing the input parameters. This
situation probably occurs because one (or more) of the following applies:
 - Your input images are very large and should be cropped to a more
   reasonable size (a large fraction of the FoV is useless). Simply use the
   parameter N=.. to crop the images appropriately.
 - The parameter N=... has been set to a value that is too large.
 - The wavelength is incorrect (too short, resulting in a small Airy pattern).
 - The f-ratio is incorrect (too small)
 - The pixel size is incorrect (too large, mind the units : meters)
 - Your images are severely undersampled: then we recommend avoiding any phase
   diversity attempt on such data (at the risk of great frustration).
"""


msg_limited_zernike_basis = """
The modal basis that will be used for retrieving the phase is Zernike modes. The
basis will range from tiptilt to Z_%d. Please do not attempt to retrieve
more than %d modes in the phase vector (a brutal crash will occur
otherwise).
"""