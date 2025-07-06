import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def symo(x, m):
    """
    Compute the values of x modulo m, in such a way that the result is comprised
    between -m/2 and m/2.

    Args:
        x (float | ndarray): value or array of values to be computed modulo m
        m (float): value of module

    Returns:
        float | ndarray: x modulo m
    """
    x = np.asarray(x)
    x = x - m * np.floor(x/m + 0.5)
    return x




def xsoft(image, alpha=0.5):
    """
    Formats the image to get ready for display with improved contrast.
    """
    im = np.fft.fftshift(image.T)
    return np.sign(im) * np.abs(im)**alpha





def test(phi_test=None, nn=64, snr=100., object_fwhm_pix=0.0, guess_modes=20, retrieved_modes=20,
         estimate_snr=True, randshift=5, defoc_z = np.array([0, -0.5e-3, 0.5e-3]), bgnd_level=0.01, K=1.0):
    """
    Test function only used for debugging and testing the Opticsetup class.
    The function creates a test image with a random phase and a random
    defocus error. The function then computes the PSF of the test image
    and adds noise to it. The function then creates an Opticsetup object
    with the test image and the defocus error. The function then calls
    the search_phase function to retrieve the phase and defocus coefficients
    from the test image. The function then displays the original and
    retrieved PSFs. The function also displays the actual phase and the
    retrieved phase. The function also prints the results.

    Args:
        phi_test (ndarray, optional): vector of modal coefficients to be used
            for the test. If None, a random vector is generated. Defaults to None.
        nn (int, optional): size of the image in [pixels]. Defaults to 64.
        snr (float, optional): signal to noise ratio. Defaults to 100.
        object_fwhm_pix (float, optional): size of the object in [pixels]. Defaults to 0.0.
        guess_modes (int, optional): number of modes to be guessed. Defaults to 20.
        retrieved_modes (int, optional): number of modes searched. Defaults to 20.
        estimate_snr (bool, optional): flag for estimating the signal to noise
            ratio and the weights in the least-square fit. Defaults to True.
        randshift (int, optional): maximum random shift in [pixels]. Defaults to 5.
        defoc_z (ndarray, optional): defocus coefficients. Defaults to
            np.array([0, -0.5e-3, 0.5e-3]).
        bgnd_level (float, optional): background level. Defaults to 0.01.
        K (float, optional): scaling factor for the random phase. Defaults to 1.0.
    Returns:
        Opticsetup: Opticsetup object with the retrieved parameters.
    """
    # initialisation for graphics
    plt.figure(1)
    plt.clf()
    # create a test image
    p = Opticsetup(np.zeros((len(defoc_z), nn, nn)), xc=None, yc=None, N=None, defoc_z=defoc_z,
                      pupilType=0, flattening=1.0, obscuration=0.12, angle=0.,
                      nedges=4, spiderAngle=0., spiderArms_m=np.full(4,0.02),
                      spiderOffset_m=np.array([1,-1,1,-1])*0.02,
                      illum=[1.0,0.1,0,-0.12,-0.152], wvl=633e-9, fratio=15.,
                      pixelSize=5e-6, edgeblur_percent=3, object_fwhm_pix=object_fwhm_pix)

    if phi_test is None:
        if guess_modes is None:
            nmodes = np.random.randint(5, 51) # Upper bound is exclusive
        else:
            nmodes = guess_modes
        phi_test = np.round(np.random.randn(nmodes)/np.sqrt(np.arange(5,5+nmodes)) * 50 * K / np.sqrt(nmodes), 2)  # Random phase coefficients
    else:
        phi_test = np.round(phi_test, 2)
    print(f'Random phase coefficients: {phi_test}')
    # compute the PSF with a random phase
    test_image = np.fft.fftshift(p.psf( phi_test ) * 1000., axes=(1,2))
    # random shift the PSFs
    dd = randshift
    for i in range(1,p.nbim):
        test_image[i] = np.roll(test_image[i], np.random.randint(-dd,dd+1,2), axis=(0,1)) # random shift
    # Add noise on the PSFs
    test_image += np.random.randn(*test_image.shape) / snr * np.max(test_image) # Add noise on the PSF
    # Add background on the PSFs
    background = np.round(np.random.randn(*defoc_z.shape) * bgnd_level * np.max(test_image),2) # Add background
    test_image += background[:,None,None] # Add background to the PSF
    # Add a random defocus error
    erreur_defoc_z = np.random.randn(*defoc_z.shape) * 1e-4 # Random defocus error
    erreur_defoc_z[0] = 0.0 # no error on the first defocus
    print(f'Random defocus error: {erreur_defoc_z*1000} mm')
    p = Opticsetup(test_image, xc=None, yc=None, N=None, defoc_z=defoc_z + erreur_defoc_z,
                      pupilType=0, flattening=1.0, obscuration=0.12, angle=0.,
                      nedges=4, spiderAngle=0., spiderArms_m=np.full(4,0.02),
                      spiderOffset_m=np.array([1,-1,1,-1])*0.02,
                      illum=[1.0,0.1,0,-0.12,-0.152], wvl=633e-9, fratio=15.,
                      pixelSize=5e-6, edgeblur_percent=3, object_fwhm_pix=object_fwhm_pix)
    p.phase = np.zeros(retrieved_modes) # initialize the phase to zero
    p.search_phase(defoc_z_flag=True, fratio_flag=False, wvl_flag=False,
                   tiptilt_flag=True, amplitude_flag=True, background_flag=True,
                   phase_flag=True, estimate_snr=estimate_snr, verbose=True, tolerance=1e-6)
    print(f'Random phase coefficients: {phi_test}')
    print(f'Retrieved coefficients   : {np.round(p.phase, 2)}')
    # print(f'Error                    : {np.round(phi_test - p.phase, 2)}')
    print(f'Random background coeffs : {background}')
    print(f'Retrieved backgr. coeffs : {np.round(p.background, 2)}')
    err_wavefront = (p.phase_generator(phi_test) - p.phase_generator(p.phase)) * p.wvl / (2*np.pi) * 1e9
    print(f'RMS error                : {np.std(err_wavefront):.2f} nm rms')
    retrieved_psf = p.psf(p.phase)
    plt.figure(3)
    plt.clf()
    plt.imshow( p.mappy(p.phase_generator(phi_test)).T * p.wvl / 2 / np.pi * 1e9, origin='lower', cmap='gray')
    plt.title('Actual Phase [nm]')
    plt.colorbar()
    # Display the original and retrieved PSFs
    plt.figure(2)
    plt.clf()
    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(p.nbim, 2, num=2, figsize=(10, 8))  # 3 rows, 2 columns
    for i in range(p.nbim):
        # Plot the PSF in the first column
        axes[i, 0].imshow(xsoft(test_image[i]), cmap='gray', origin='lower')
        axes[i, 0].set_title(f"PSF {i+1}")
        axes[i, 0].axis('off')
        # Plot the other PSF in the second column
        axes[i, 1].imshow(xsoft(retrieved_psf[i]), cmap='gray', origin='lower')
        axes[i, 1].set_title(f"Retrieved PSF {i+1}")
        axes[i, 1].axis('off')
    # Adjust layout
    fig.tight_layout()
    return p



