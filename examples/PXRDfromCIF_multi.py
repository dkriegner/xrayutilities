import os
import numpy as np
import xrayutilities as xu
import copy
from xrayutilities.materials.spacegrouplattice import WyckoffBase
from multiprocessing import freeze_support
from enum import Enum
from typing import List, Tuple, Dict


class Shape(Enum):
    Gaussian = 1
    Lorentzian = 2
    Neither = 3

class Diffractogram:
    def __init__(self, lambda_used, two_theta, shape):
        self.lambda_used = lambda_used
        self.two_theta = two_theta
        self.shape = shape
        self.cifs = []
        self.name_sol = []
        self.concentration_sol = []
        self.cryst_size = []
        self.intensity = []
        self.vol_per_atom = []
        self.concentration_coex = np.array([], dtype=object)
        self.intensity_coex = np.zeros(len(self.two_theta))

    def add_phase(self, cif: List[os.PathLike], cryst_size):
        self.cifs.append(np.array(cif))
        self.concentration_sol.append(np.array([1.0])) # at%
        self.name_sol.append('')
        self.cryst_size.append(cryst_size) # meter
    
    def compute_intensity(self, concentration_coex):
        self.concentration_coex = np.array(concentration_coex)

        # create solution phase cif
        def needs_solution_phase_cif(lof_cifs):
            return len(lof_cifs) > 1

        # check input
        if len(self.concentration_coex) != len(self.cifs) or np.sum(self.concentration_coex) != 1.0:
            raise ValueError("Per coexisting phase a concentration has to be specified. All concentrations have to sum up to 1.0.")    

        for i, cif_files in enumerate(self.cifs):
            if needs_solution_phase_cif(self.cifs[i]):
                if len(self.concentration_sol[i]) != len(self.cifs[i]) or np.sum(self.concentration_sol[i]) != 1.0:
                    raise ValueError("A concentration [at%] per end member has to be specified. All concentrations have to sum up to 1.0.")
                if self.name_sol[i] == "":
                    raise ValueError("Please specify a name to save the solution phase cif file.")
            else:
                if len(self.concentration_sol[i]) != 1 or self.concentration_sol[i][0] != 1.0:
                    raise ValueError("For pure coexisting phases, only one concentration of 1.0 has to be specified.")
                if self.name_sol[i] != "":
                    raise ValueError("No need to specify name for pure coexisting phase since no new cif file will be created.")
                
            # compute
            if needs_solution_phase_cif(self.cifs[i]):
                cif_file_sol = create_sol_phase(cif_files, concentration_sol=self.concentration_sol[i], name_sol=self.name_sol[i])
            else:
                cif_file_sol =  cif_files[0]

            inte_pdf, vol_pdf = cif_to_diffractogram(cif_file = cif_file_sol, lambda_used=self.lambda_used, shape=self.shape, cryst_size=self.cryst_size[i], two_theta=self.two_theta)
            self.intensity.append(inte_pdf)
            self.vol_per_atom.append(vol_pdf)

        # calculate intensity for coexisting phases
        self.intensity_coex = combine_intensities(self.intensity, self.vol_per_atom, self.concentration_coex)
    
    def plot_diffractogram_matplotlib(self, **kwargs):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.two_theta, self.intensity_coex)
        plt.xlabel(kwargs.get("xlabel","2θ/degrees"))
        plt.ylabel(kwargs.get("ylabel","Intensity"))
        plt.gca().axes.yaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.title(kwargs.get("title","Calculated Powder Diffraction Pattern"))
        plt.show(block=False)

def create_sol_phase(cif_files_list:List, concentration, name_sol: str = "last.cif"):
    # create array for phases in solution phase
    sol_phase = []
    cif_files = np.array(cif_files_list)
    concentration_sol = np.array(concentration)

    # create material
    for cif_file_comp in cif_files:
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
        material_sol.a += concentration_sol[j] * sol_phase[j].a
        material_sol.b += concentration_sol[j] * sol_phase[j].b
        material_sol.c += concentration_sol[j] * sol_phase[j].c

        for atom, wyckoff, occ, b in sol_phase[j].lattice._wbase:
            new_occ = float(concentration_sol[j])  # Replace occupancy with concentration
            new_wbase.append(atom, wyckoff, occ=new_occ, b=b)

    # correct for 1.0
    material_sol.a += -1.0
    material_sol.b += -1.0
    material_sol.c += -1.0

    # Assign the new WyckoffBase to the lattice
    material_sol.lattice._wbase = new_wbase

    # export to cif file
    material_sol.toCIF(os.path.join("cif", name_sol))

    return name_sol


def cif_to_diffractogram(
    cif_file: str, lambda_used=1.5406, shape=Shape.Gaussian, cryst_size=1e-7, two_theta=np.linspace(10, 135, 500)
):
    # acknoledge processing
    print(f"------------------------- Currently processing information from {cif_file} -------------------------")

    # create material
    cif_path = os.path.join("cif", cif_file)
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"The CIF file '{cif_file}' does not exist in the 'cif' folder.")
    material = xu.materials.Crystal.fromCIF(cif_path)

    # create pdf file
    powder_cal = xu.simpack.PowderDiffraction(material, enable_simulation=True)

    # change wavelength to selected
    powder_cal.wavelength = lambda_used

    # change peak shape
    match shape:
        case Shape.Gaussian:
            powder_cal.settings["emission"]["crystallite_size_gauss"] = cryst_size
            powder_cal.settings["emission"]["crystallite_size_lor"] = 1e10
        case Shape.Lorentzian:
            powder_cal.settings["emission"]["crystallite_size_gauss"] = 1e10
            powder_cal.settings["emission"]["crystallite_size_lor"] = cryst_size
        case Shape.Neither:
            powder_cal.settings["emission"]["crystallite_size_gauss"] = 1e10
            powder_cal.settings["emission"]["crystallite_size_lor"] = 1e10

    powder_cal.update_settings({"emission": powder_cal.settings["emission"]})

    # update pdf
    powder_cal.set_sample_parameters()
    powder_cal.update_powder_lines(tt_cutoff=two_theta[-1])
    print(powder_cal)

    # calculate weight factor and the intensity for diffractogram of this phase
    V_at = material.lattice.UnitCellVolume() / len(material.lattice._wbase)
    I = powder_cal.Calculate(two_theta)

    return I, V_at


def combine_intensities(intensity, vol_per_atom, concentration_coex):
    vol_tot = np.sum(vol_per_atom * concentration_coex)

    # apply weight factor to intensity
    intensity_all = np.zeros(len(intensity[0][:]))
    for i, phase in enumerate(intensity):
        intensity_all += intensity[i][:] * (concentration_coex[i] * vol_per_atom[i] / vol_tot)

    return intensity_all

if __name__ == "__main__":

    # Example case 1
    diff_1 = Diffractogram(lambda_used=1.5406, two_theta=np.linspace(10, 135, 1000), shape=Shape.Gaussian)

    diff_1.add_phase(["Fe.cif"], cryst_size=1e-7)
    new_cif_sol = create_sol_phase(["Fe.cif", "Ni.cif"], concentration=[0.3,0.7], name_sol="FeNi_sol.cif")
    diff_1.add_phase(new_cif_sol, cryst_size=1e-7)
    diff_1.compute_intensity(concentration_coex=[0.3, 0.7])
    diff_1.plot_diffractogram_matplotlib()

    # Example 2
    diff_2 = Diffractogram(lambda_used=1.5406, two_theta=np.linspace(10, 135, 1000), shape=Shape.Lorentzian)

    diff_2.add_phase(["Fe.cif"], cryst_size=1e-7)
    diff_2.add_phase(["Ni.cif"], cryst_size=1e-7)
    diff_2.compute_intensity(concentration_coex=[0.5, 0.5])
    diff_2.plot_diffractogram_matplotlib()
