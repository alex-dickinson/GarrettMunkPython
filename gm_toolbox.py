import numpy as np

# import numpy.ma as ma
import numpy.matlib as matlib
import matplotlib.pyplot as plt

# Python implementation of part of Jody Klymak's Matlab Garrett-Munk interval wave spectra toolbox (see http://jklymak.github.io/GarrettMunkMatlab). These Python scripts compute the GM76 and GK91 model spectra


def vertical_spectrum_Akz(kz, jstar, N, N0, b, gm_form, gm_rolloff_option):

    kzstar = jstar * (N / N0) / (2.0 * b)

    # Choose GM form
    if gm_form == "gm75":
        I = 1.5
        A_no_rolloff_term = (1.0 / kzstar) / np.power((1. + (kz/kzstar)), 2.5)
    elif gm_form == "gm76":
        I = 2.0 / np.pi
        A_no_rolloff_term = (1.0 / kzstar) / (1.0 + np.power((kz / kzstar), 2.0))
    elif gm_form == "gk91":
        I = 1.0
        A_no_rolloff_term = (1.0 / kzstar) / np.power((1.0 + (kz / kzstar)), 2.0)
    else:
        print("gm_option not specified - exiting")
        exit()
   
    A_rolloff_term = ((1.0 / kzstar) / (1.0 + np.power((0.1 / kzstar), 2.0)) * np.power((kz / 0.1), -3.0))

    # Set up saturated subrange if specified
    if gm_rolloff_option == "yes":
        A = I * np.minimum(A_no_rolloff_term, A_rolloff_term)
    elif gm_rolloff_option == "no":
        A = I * A_no_rolloff_term
    else:
        print("gm_rolloff_option not specified - exiting")
        exit()

    return A


# --------------------------------------------------------------------------------


def frequency_spectrum_Bomega(omega, f):

    B = (
        (2.0 / np.pi)
        * f
        / (omega * np.power((np.power(omega, 2.0) - np.power(f, 2.0)), 0.5))
    )

    return B


# --------------------------------------------------------------------------------


def make_kxkz_displacement_spectra(
    kz, kx, N, N0, f, NkH, E, b, jstar, gm_rolloff_option
):

    ### Set up matrix S for kxkz spectrum
    S = np.zeros([np.size(kz), np.size(kx)])

    ### Iterate over values of kx
    for i in range(np.size(kx)):

        ### Set array of values for integration over total horizontal wavenumber, kH.
        NkH_array = np.arange(0, NkH, 1)
        Z = np.zeros((np.size(NkH_array), np.size(kz)))
        for j in range(np.size(kz)):
            Z[:, j] = NkH_array
        Z = Z / (NkH - 1)

        ### Set up matrix of kz values
        Kz = matlib.repmat(kz, np.size(Z, 0), 1)

        ### Scale Z matrix to values required by integration over kH.
        zmax = np.power(((np.power(Kz, 2.0) / np.power(kx[i], 2.0)) - 1.0), 0.5)
        Z = Z * zmax

        ### Set up frequency matrix
        omsq = np.power(kx[i], 2.0) / np.power(Kz, 2.0) * (np.power(Z, 2.0) + 1.0) * (
            np.power(N, 2.0) - np.power(f, 2.0)
        ) + np.power(f, 2.0)
        om = np.power(omsq, 0.5)

        ### Calculate derivative of frequency with respect to horizontal wavenumber
        domda = (
            kx[i]
            * np.power((np.power(Z, 2.0) + 1.0), 0.5)
            / om
            / np.power(Kz, 2.0)
            * (np.power(N, 2.0) - np.power(f, 2.0))
        )

        ### Calculate increment for integration over vertical and total horizontal wavenumber
        dz = matlib.repmat(np.diff(Z[0:2], axis=0), np.size(Z, 0), 1)
        Tda = 1.0 / np.power((np.power(Z, 2.0) + 1.0), 0.5) * dz

        ### Mean-square quantity for displacement
        R = (
            (1.0 / np.power(N, 2.0))
            * (np.power(om, 2.0) - np.power(f, 2.0))
            / np.power(om, 2.0)
        )

        ### Dimensional energy density
        E0 = np.power(b, 2.0) * np.power(N0, 2.0) * E

        ### Calculate vertical spectrum for given frequency and kH values
        A = vertical_spectrum_Akz(Kz, jstar, N, N0, b, gm_form, gm_rolloff_option)

        ### Calculate frequency spectrum
        B = frequency_spectrum_Bomega(om, f)

        ### Combine elements calculated above to yield kxkz spectrum (for given value of kx) in terms of kz and omega
        TT = E0 * (N / N0) * R * A * B * Tda * domda

        ### Integrate over omega
        S[:, i] = np.trapz(TT, axis=0)

    return S


# --------------------------------------------------------------------------------


def integrate_over_kx(S, kx, kz):

    Snan = S
    S = np.nan_to_num(S)
    Skz = np.trapz(S, x=kx, axis=1)

    ### Multiply by final factor of 2/pi. What is this for?
    Skz = 2.0 / np.pi * Skz

    return Skz


# --------------------------------------------------------------------------------


def integrate_over_kz(S, kx, kz):

    Snan = S
    S = np.nan_to_num(S)
    Skx = np.trapz(S, x=kz, axis=0)

    ### Multiply by final factor of 2/pi. What is this for?
    Skx = 2.0 / np.pi * Skx

    return Skx


# --------------------------------------------------------------------------------

def compute_kx_from_kxkz(N_rads, N0_rads, f_rads, Nkz, kz, Nkx, kx, NkH, E, b, jstar, gm_form, gm_rolloff_option):
    
    
    

    ###### Run functions
    ### Derive kx and kz displacement spectra by integrating kxkz spectrum
    kxkz_displacement_spectrum = make_kxkz_displacement_spectra(
        kz, kx, N_rads, N0_rads, f_rads, NkH, E, b, jstar, gm_rolloff_option
    )
    # 	kz_displacement_spectrum = integrate_over_kx(kxkz_displacement_spectrum, kx, kz)
    # 	kz_displacement_z_spectrum = np.power((2.*np.pi*kz), 2.) * kz_displacement_spectrum
    kx_displacement_spectrum = integrate_over_kz(kxkz_displacement_spectrum, kx, kz)
    kx_displacement_x_spectrum = (
        np.power((2.0 * np.pi * kx), 2.0) * kx_displacement_spectrum
    )

    # 	plt.loglog(kx, kx_displacement_spectrum)
    # 	plt.plot(np.log10(kx), np.log10(kx_displacement_x_spectrum))
    # 	plt.loglog(kx, kx_displacement_x_spectrum)
    # 	plt.show()

    # 	exit()

    # outwrite.write("> -Z %f" % N_cph)
    # for n in range(np.size(kx)):
    # 	outwrite.write("%f %f %f %f %f %f\n" % (kx[n], np.log10(kx[n]), kx_displacement_spectrum[n], np.log10(kx_displacement_spectrum[n]), kx_displacement_x_spectrum[n], np.log10(kx_displacement_x_spectrum[n])))

    # individual_outfile = "./temp_gm_data/kx_displacement_spectra_N" + str(N_cph) + "cph_python.dat"

    # 	np.savetxt(gm_spectrum_outfile, np.column_stack((kx, np.log10(kx), kx_displacement_spectrum, np.log10(kx_displacement_spectrum), kx_displacement_x_spectrum, np.log10(kx_displacement_x_spectrum))))

    return (
        np.log10(kx),
        np.log10(kx_displacement_spectrum),
        np.log10(kx_displacement_x_spectrum),
    )