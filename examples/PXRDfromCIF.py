import os
import numpy as np
import xrayutilities as xu
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def main():

    # --------------- START: ToDo ---------------
    # cif file to load
    cif = "FeNi3.cif"

    # wavelength
    lambda_used = 1.5406

    # two theta range
    two_theta = np.linspace(10, 135, 1000)

    # peak shape
    shape = 1 # 0: neiter, 1: gaussian, 2: lorentzian
    cryst_size = 1e-7
    # --------------- END: ToDo ---------------

    # create material
    material = xu.materials.Crystal.fromCIF(os.path.join("cif", cif))

    # create pdf file
    powder_cal = xu.simpack.PowderDiffraction(material, enable_simulation=True)

    # change wavelength to selected
    powder_cal.wavelength = lambda_used

    # change peak shape
    if shape == 0:
        powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10
        powder_cal.settings['emission']['crystallite_size_lor'] = 1e10
    elif shape == 1:
        powder_cal.settings['emission']['crystallite_size_gauss'] = cryst_size
        powder_cal.settings['emission']['crystallite_size_lor'] = 1e10 
    elif shape == 2:
        powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10 
        powder_cal.settings['emission']['crystallite_size_lor'] = cryst_size 
    powder_cal.update_settings({'emission': powder_cal.settings['emission']})

    # update pdf
    powder_cal.set_sample_parameters()
    powder_cal.update_powder_lines(tt_cutoff=two_theta[-1])
    print(powder_cal)

    # calculate the diffractogram
    intensity = powder_cal.Calculate(two_theta)

    # plot the result
    plt.plot(two_theta, intensity)
    plt.xlabel("2θ/degrees")
    plt.ylabel("Intensity")
    plt.gca().axes.yaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.title(f"Powder Diffraction Pattern of {cif}")
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()