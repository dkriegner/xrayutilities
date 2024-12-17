import os
import numpy as np
import xrayutilities as xu
from xrayutilities.materials.spacegrouplattice import WyckoffBase
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def main():
    # initializing lists
    cifs = []
    name_sol = []
    concentration_sol = []
    cryst_size = []

    # --------------- START: ToDo ---------------
    # Coexisting phase 0
    cifs.append(np.array(["Fe.cif", "Ni.cif"])) # cif file(s) to load for each coexisting phase (each phase needs one cif file, specify two cif files if it is a solution phase)
    concentration_sol.append(np.array([0.5, 0.5])) # if solution phase: specify concentration of each constituent, otherwise 1.0
    name_sol.append('FeNi_fcc_sol.cif') # if solution phase: specify name for cif, otherwise ''
    cryst_size.append(1e-7) # meter (one value per phase)

    # Coexisting phase 1
    cifs.append(np.array(["FeNi_fcc.cif"])) # cif file(s) to load for each coexisting phase (each phase needs one cif file, specify two cif files if it is a solution phase)
    concentration_sol.append(np.array([1.0])) # if solution phase: specify concentration of each constituent, otherwise 1.0
    name_sol.append('') # if solution phase: specify name for cif, otherwise ''
    cryst_size.append(1e-7) # meter (one value per phase)

    # specify concentration of coexisting phases, otherwise specify 1.0
    concentration_coex = np.array([0.9, 0.1])

    # wavelength
    lambda_used = 1.5406 # Angström

    # two theta range
    two_theta = np.linspace(10, 135, 1000)

    # peak shape
    shape = 1 # 0: neither, 1: gaussian, 2: lorentzian
    # --------------- END: ToDo ---------------

    # flag for each coexisting phase, 0: seperate display or 1: solution phase
    sol = np.zeros(len(cifs))

    # intensity and volume weight factor
    intensity = np.zeros(len(two_theta), len(cifs))
    vol_at = np.zeros(len(cifs))
    vol_tot = 0

    # set solution phase flag and check for value errors
    coex_sum = 0
    for i, cif_files in enumerate(cifs):
        coex_sum = coex_sum + concentration_coex[i]
        if len(cif_files) > 1:
            sol[i] = 1
            if len(concentration_sol[i]) != 2:
                raise ValueError("A concentration [at%] per end member has to be specified.")
            if name_sol[i] == '':
                raise ValueError("Please specify a name to save the solution phase cif file.")
            if concentration_sol[i][0] + concentration_sol[i][0] != 1.0:
                raise ValueError("The concentrations [at%] for each solution phase have to sum up to 1.0.")
        else:
            sol[i] = 0
            if len(concentration_sol[i]) != 1 or concentration_sol[i][0] != 1.0:
                raise ValueError("For pure coexisting phases, only one concentration of 1.0 has to be specified.")
            if name_sol[i] != '':
                raise ValueError("No need to specify name for pure coexisting phase since no new cif file will be created.")  
    if len(concentration_coex) != len(cifs) or coex_sum != 1.0:
        raise ValueError("Per coexisting phase a concentration has to be specified. All concentrations have to sum up to 1.0.")
    
    # calculate powder diffractograms
    for i, cif_files in enumerate(cifs):

        if sol[i] == 0:
            # create material
            material = xu.materials.Crystal.fromCIF(os.path.join("cif", cif_files))

            # create pdf file
            powder_cal = xu.simpack.PowderDiffraction(material, enable_simulation=True)

            # change wavelength to selected
            powder_cal.wavelength = lambda_used

            # change peak shape
            if shape == 1:
                powder_cal.settings['emission']['crystallite_size_gauss'] = cryst_size[i]
                powder_cal.settings['emission']['crystallite_size_lor'] = 1e10 
            elif shape == 2:
                powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10 
                powder_cal.settings['emission']['crystallite_size_lor'] = cryst_size[i]
            else:
                powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10
                powder_cal.settings['emission']['crystallite_size_lor'] = 1e10
                if shape != 0:
                    print('Warning! The peak shape value was set to 0 and is not neither Gaussian nor Lorentzian.')
            powder_cal.update_settings({'emission': powder_cal.settings['emission']})

            # update pdf
            powder_cal.set_sample_parameters()
            powder_cal.update_powder_lines(tt_cutoff=two_theta[-1])
            print(f"------------------------- Currently processing information from {cif_files} -------------------------")
            print(powder_cal)

            # calculate weight factor and the intensity for diffractogram of this phase
            vol_at[i] = material.lattice.UnitCellVolume() / len(material.lattice._wbase)
            vol_tot += concentration_coex[i] * vol_at[i]
            intensity[:,i] = powder_cal.Calculate(two_theta)

        elif sol[i] == 1:
            # create array for phases in solution phase
            sol_phase = []

            # create material
            for cif_file in cif_files:
                sol_phase.append(xu.materials.Crystal.fromCIF(os.path.join("cif", cif_file)))

            # adapt lattice parameter
            material_sol = sol_phase[0]
            material_sol.a = 0
            material_sol.b = 0
            material_sol.c = 0

            # adapt wyckoffBase
            new_wbase = WyckoffBase()

            for j in range(len(sol_phase)):
                material_sol.a += concentration_sol[j] * sol_phase[j].a
                material_sol.b += concentration_sol[j] * sol_phase[j].b
                material_sol.c += concentration_sol[j] * sol_phase[j].c

                for atom, wyckoff, occ, b in sol_phase[j].lattice._wbase:
                    new_occ = float(concentration_sol[j])  # Replace occupancy with concentration
                    new_wbase.append(atom, wyckoff, occ=new_occ, b=b)

            # Assign the new WyckoffBase to the lattice
            material_sol.lattice._wbase = new_wbase

            # export to cif file
            material_sol.toCIF(os.path.join("cif", name_sol[i]))
            
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
            print(f"------------------------- Currently processing information from {name_sol[i]} -------------------------")
            print(powder_cal)

            # calculate weight factor and the intensity for diffractogram of this phase
            vol_at[i] = material_sol.lattice.UnitCellVolume() / len(material_sol.lattice._wbase)
            vol_tot += concentration_coex[i] * vol_at[i]
            intensity[:,i] = powder_cal.Calculate(two_theta)

    # apply weight factor to intensity
    intensity_all = 0
    for i, cif_files in enumerate(cifs):
        intensity_all += intensity[:,i] * ( concentration_coex[i] * vol_at[i] / vol_tot )

    # plotting
    plt.plot(two_theta, intensity_all)
    plt.xlabel("2θ/degrees")
    plt.ylabel("Intensity")
    plt.gca().axes.yaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.title(f"Powder Diffraction Pattern(s)")
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()