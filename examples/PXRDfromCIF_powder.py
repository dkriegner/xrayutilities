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

    # create material
    material = xu.materials.Crystal.fromCIF(os.path.join("cif", cif))

    # two theta range
    two_theta = np.linspace(40, 65, 300)

    # peak shape
    shape = 1 # 1: gaussian, 2: lorentzian
    cryst_size = 1e-10
    # --------------- END: ToDo ---------------

    # create powder
    if shape == 1:
        powder = xu.simpack.Powder(material, 1, crystallite_size_gauss=cryst_size, crystallite_size_lor=1e10)
    elif shape == 2:
        powder = xu.simpack.Powder(material, 1, crystallite_size_gauss=1e10, crystallite_size_lor=cryst_size)
    pm = xu.simpack.PowderModel(powder)

    # # update wavelength
    # pm.pdiff.settings['emission']['emiss_wavelengths'] = (lambda_used,)
    # pm.pdiff.update_settings({'emission': pm.pdiff[0].settings['emission']})
    
    # plot the result
    pm.simulate(two_theta)
    ax = pm.plot(two_theta)
    ax.set_xlim(two_theta[0], two_theta[-1])
    plt.show()
    pm.close()

if __name__ == '__main__':
    freeze_support()
    main()