import os
import numpy as np
import xrayutilities as xu
import copy
from xrayutilities.materials.spacegrouplattice import WyckoffBase
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from enum import Enum
from typing import List, Tuple, Dict

class Shape(Enum):
    Gaussian = 1
    Lorentzian = 2
    Neither = 3

def create_sol_phase(_cif_files: np.ndarray, _concentration_sol, _name_sol: str = 'last.cif'):
    # create array for phases in solution phase
    sol_phase = []

    # create material
    for cif_file_comp in _cif_files:
        cif_path = os.path.join("cif", cif_file_comp)
        if not os.path.exists(cif_path):
            raise FileNotFoundError(f"The CIF file '{cif_file_comp}' does not exist in the 'cif' folder.")
        
        sol_phase.append(xu.materials.Crystal.fromCIF(cif_path))    

    # adapt lattice parameter
    material_sol = copy.deepcopy(sol_phase[0])
    material_sol.a = 1.0
    material_sol.b = 1.0
    material_sol.c = 1.0

    # adapt wyckoffBase
    new_wbase = WyckoffBase()

    for j in range(len(sol_phase)):
        material_sol.a += _concentration_sol[j] * sol_phase[j].a
        material_sol.b += _concentration_sol[j] * sol_phase[j].b
        material_sol.c += _concentration_sol[j] * sol_phase[j].c

        for atom, wyckoff, occ, b in sol_phase[j].lattice._wbase:
            new_occ = float(_concentration_sol[j])  # Replace occupancy with concentration
            new_wbase.append(atom, wyckoff, occ=new_occ, b=b)

    # correct for 1.0
    material_sol.a += -1.0
    material_sol.b += -1.0
    material_sol.c += -1.0

    # Assign the new WyckoffBase to the lattice
    material_sol.lattice._wbase = new_wbase

    # export to cif file
    material_sol.toCIF(os.path.join("cif", _name_sol))

    return _name_sol

def cif_to_diffractogram(_cif_file: str, _lambda_used=1.5406, _shape=Shape.Gaussian, _cryst_size=1e-7, _two_theta=np.linspace(10, 135, 500)):
    # acknoledge processing
    print(f"------------------------- Currently processing information from {_cif_file} -------------------------")

    # create material
    cif_path = os.path.join("cif", _cif_file)
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"The CIF file '{_cif_file}' does not exist in the 'cif' folder.")      
    material = xu.materials.Crystal.fromCIF(cif_path)

    # create pdf file
    powder_cal = xu.simpack.PowderDiffraction(material, enable_simulation=True)

    # change wavelength to selected
    powder_cal.wavelength = _lambda_used

    # change peak shape
    match _shape:
        case Shape.Gaussian:
            powder_cal.settings['emission']['crystallite_size_gauss'] = _cryst_size
            powder_cal.settings['emission']['crystallite_size_lor'] = 1e10 
        case Shape.Lorentzian:
            powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10 
            powder_cal.settings['emission']['crystallite_size_lor'] = _cryst_size
        case Shape.Neither:
            powder_cal.settings['emission']['crystallite_size_gauss'] = 1e10
            powder_cal.settings['emission']['crystallite_size_lor'] = 1e10

    powder_cal.update_settings({'emission': powder_cal.settings['emission']})

    # update pdf
    powder_cal.set_sample_parameters()
    powder_cal.update_powder_lines(tt_cutoff=_two_theta[-1])
    print(powder_cal)

    # calculate weight factor and the intensity for diffractogram of this phase
    V_at = material.lattice.UnitCellVolume() / len(material.lattice._wbase)
    I = powder_cal.Calculate(_two_theta)

    return I, V_at

def combine_diffractograms(_intensity, _vol_per_atom, _concentration_coex, _vol_tot):

    # apply weight factor to intensity
    intensity_all = np.zeros(len(_intensity[0][:]))
    for i, phase in enumerate(_intensity):
        intensity_all += _intensity[i][:] * ( _concentration_coex[i] * _vol_per_atom[i] / _vol_tot )
    
    return intensity_all

def plot_diffractogram(_intensity_coex, _two_theta):

    plt.plot(_two_theta, _intensity_coex)
    plt.xlabel("2θ/degrees")
    plt.ylabel("Intensity")
    plt.gca().axes.yaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.title(f"Calculated Powder Diffraction Pattern")
    plt.show()

def main():

    # initializing lists
    cifs = []
    name_sol = []
    concentration_sol = []
    cryst_size = []
    intensity = []
    vol_per_atom = []

    # --------------- START: ToDo ---------------
    # Coexisting phase 0
    cifs.append(np.array(["Ni.cif", "Fe.cif"])) # cif file(s) to load for each coexisting phase (each phase needs one cif file, specify two cif files if it is a solution phase)
    concentration_sol.append(np.array([0.7,0.3])) # if solution phase: specify concentration of each constituent, otherwise 1.0
    name_sol.append('Anna.cif') # if solution phase: specify name for cif, otherwise ''
    cryst_size.append(1e-7) # meter (one value per phase)

    # Coexisting phase 1
    cifs.append(np.array(["Fe.cif"])) # cif file(s) to load for each coexisting phase (each phase needs one cif file, specify two cif files if it is a solution phase)
    concentration_sol.append(np.array([1.0])) # if solution phase: specify concentration of each constituent, otherwise 1.0
    name_sol.append('') # if solution phase: specify name for cif, otherwise ''
    cryst_size.append(1e-7) # meter (one value per phase)

    # specify concentration of coexisting phases, otherwise specify 1.0
    concentration_coex = np.array([0.3, 0.7])

    # wavelength
    lambda_used = 1.5406 # Angström

    # two theta range
    two_theta = np.linspace(10, 135, 1000)
    # --------------- END: ToDo ---------------

    # set solution phase flag and check for value errors
    if len(concentration_coex) != len(cifs) or np.sum(concentration_coex) != 1.0:
        raise ValueError("Per coexisting phase a concentration has to be specified. All concentrations have to sum up to 1.0.")     

    for i, cif_files in enumerate(cifs):
        
        if len(cifs[i]) > 1:
            if len(concentration_sol[i]) != len(cifs[i]) or np.sum(concentration_sol[i]) != 1.0:
                raise ValueError("A concentration [at%] per end member has to be specified. All concentrations have to sum up to 1.0.")
            if name_sol[i] == '':
                raise ValueError("Please specify a name to save the solution phase cif file.")
        else:
            if len(concentration_sol[i]) != 1 or concentration_sol[i][0] != 1.0:
                raise ValueError("For pure coexisting phases, only one concentration of 1.0 has to be specified.")
            if name_sol[i] != '':
                raise ValueError("No need to specify name for pure coexisting phase since no new cif file will be created.")  
    
        # create solution phase cif
        if len(cifs[i]) > 1:
            cif_file_sol = create_sol_phase(cif_files, _concentration_sol=concentration_sol[i], _name_sol=name_sol[i])
            inte_pdf, vol_pdf = cif_to_diffractogram(cif_file_sol, _lambda_used=lambda_used, _shape=Shape.Gaussian, _cryst_size=cryst_size[i], _two_theta=two_theta)
        else:
            inte_pdf, vol_pdf = cif_to_diffractogram(cif_files[0], _lambda_used=lambda_used, _shape=Shape.Gaussian, _cryst_size=cryst_size[i], _two_theta=two_theta)

        # set intensity and volume per atom for each cif
        intensity.append(inte_pdf)
        vol_per_atom.append(vol_pdf)

    # calculate intensity for coexisting phases
    vol_tot = np.sum(vol_per_atom * concentration_coex)
    intensity_coex = combine_diffractograms(intensity, vol_per_atom, concentration_coex, vol_tot)

    # plot it
    plot_diffractogram(intensity_coex, two_theta)

if __name__ == '__main__':
    freeze_support()
    main()