import os
import re
import numpy as np
import xrayutilities as xu
from xrayutilities.materials.spacegrouplattice import WyckoffBase
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def main():

    # --------------- START: ToDo ---------------
    # cif file(s) to load (needs to be two for one solution phase, otherwise any number of cif files)
    cif = np.array(["Fe.cif", "Ni.cif"])

    # flag, 0: seperate display or 1: solution phase
    sol = 0

    # if sol == 1, put name of solution phase cif and concentration [at%] of end members here, otherwise irrelevant
    name_sol = 'FeNi_fcc.cif'
    concentration = np.array([0.3, 0.7])

    # wavelength
    lambda_used = 1.5406 # Angström

    # two theta range
    two_theta = np.linspace(10, 135, 1000)

    # peak shape
    shape = 1 # 0: neither, 1: gaussian, 2: lorentzian
    cryst_size = np.array([1e-7]) # meter (one value per cif, but one value if solution phase)
    # --------------- END: ToDo ---------------

    # for solution phases
    if sol == 1:
        # check input
        if len(cif) != 2:
            raise ValueError("Two seperate cif files are needed for the construction of a solution phase.")
        if len(cryst_size) != 1:
            raise ValueError("For a solution phase only one crystallite size value is required.")
        if len(concentration) != 2:
            raise ValueError("A concentration [at%] per end member has to be specified.")
        if concentration[0] + concentration[1] != 1.0:
            raise ValueError("Both concentrations [at%] have to sum up to 1.0.")
        
    # for seperatre phases
    elif sol == 0:
        if len(cif) != len(cryst_size):
            # check cif and crystal size inputs
            raise ValueError(f"Arrays 'cif' and 'cryst_size' must have the same length. Got len(cif)={len(cif)} and len(cryst_size)={len(cryst_size)}.")
        
    # check solution phase input
    else:
        raise ValueError("The solution phase flag has to be set either to 0 (seperate display) or 1 (solution phase).")

    if sol == 0:

        # create intensity vector
        intensity = np.zeros((len(two_theta), len(cif)))

        for cry in range(len(cif)):
            # create material
            material = xu.materials.Crystal.fromCIF(os.path.join("cif", cif[cry]))

            # create pdf file
            powder_cal = xu.simpack.PowderDiffraction(material, enable_simulation=True)

            # change wavelength to selected
            powder_cal.wavelength = lambda_used

            # change peak shape
            if shape == 1:
                powder_cal.settings['emission']['crystallite_size_gauss'] = cryst_size[cry]
                powder_cal.settings['emission']['crystallite_size_lor'] = 1e10 
            elif shape == 2:
                powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10 
                powder_cal.settings['emission']['crystallite_size_lor'] = cryst_size[cry]
            else:
                powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10
                powder_cal.settings['emission']['crystallite_size_lor'] = 1e10
                if shape != 0:
                    print('Warning! The peak shape value was set to 0 and is not neither Gaussian nor Lorentzian.')
            powder_cal.update_settings({'emission': powder_cal.settings['emission']})

            # update pdf
            powder_cal.set_sample_parameters()
            powder_cal.update_powder_lines(tt_cutoff=two_theta[-1])
            print(f"------------------------- Currently processing information from {cif[cry]} -------------------------")
            print(powder_cal)

            # calculate the diffractogram for seperate display
            intensity[:,cry] = powder_cal.Calculate(two_theta)
    
    elif sol == 1:
            # create intensity vector
            intensity = np.zeros((len(two_theta), 1))

            # create material
            material_0 = xu.materials.Crystal.fromCIF(os.path.join("cif", cif[0]))
            material_1 = xu.materials.Crystal.fromCIF(os.path.join("cif", cif[1]))

            # adapt lattice parameter
            material_sol = material_0
            material_sol.a = concentration[0] * material_0.a + concentration[1] * material_1.a
            material_sol.b = concentration[0] * material_0.b + concentration[1] * material_1.b
            material_sol.c = concentration[0] * material_0.c + concentration[1] * material_1.c

            # adapt wyckoffBase
            new_wbase = WyckoffBase()
            # Update lines from material_0
            for atom, wyckoff, occ, b in material_0.lattice._wbase:
                new_occ = float(concentration[0])  # Replace occupancy with concentration[0]
                new_wbase.append(atom, wyckoff, occ=new_occ, b=b)
            # Update lines from material_1
            for atom, wyckoff, occ, b in material_1.lattice._wbase:
                new_occ = float(concentration[1])  # Replace occupancy with concentration[1]
                new_wbase.append(atom, wyckoff, occ=new_occ, b=b)

            # Assign the new WyckoffBase to the lattice
            material_sol.lattice._wbase = new_wbase

            # export to cif file
            material_sol.toCIF(os.path.join("cif", name_sol))

            # create pdf file
            powder_cal = xu.simpack.PowderDiffraction(material_sol, enable_simulation=True)

            # change wavelength to selected
            powder_cal.wavelength = lambda_used

            # change peak shape
            if shape == 1:
                powder_cal.settings['emission']['crystallite_size_gauss'] = cryst_size[0]
                powder_cal.settings['emission']['crystallite_size_lor'] = 1e10 
            elif shape == 2:
                powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10 
                powder_cal.settings['emission']['crystallite_size_lor'] = cryst_size[0]
            else:
                powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10
                powder_cal.settings['emission']['crystallite_size_lor'] = 1e10
                if shape != 0:
                    print('Warning! The peak shape value was set to 0 and is not neither Gaussian nor Lorentzian.')
            powder_cal.update_settings({'emission': powder_cal.settings['emission']})

            # update pdf
            powder_cal.set_sample_parameters()
            powder_cal.update_powder_lines(tt_cutoff=two_theta[-1])
            print(powder_cal)

            # calculate the diffractogram for seperate display
            intensity[:,0] = powder_cal.Calculate(two_theta)

    # plot settings
    if sol == 0:
        for cry in range(len(cif)):
            plt.plot(two_theta, intensity[:,cry], label=f"{cif[cry]}")
    elif sol == 1:
        plt.plot(two_theta, intensity[:,0], label=f"Sol. phase of {cif[0]} and {cif[1]}, ratio: {concentration[0]}/{concentration[1]}")
    plt.xlabel("2θ/degrees")
    plt.ylabel("Intensity")
    plt.gca().axes.yaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.legend()
    plt.title(f"Powder Diffraction Pattern(s)")
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()