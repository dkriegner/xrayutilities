import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from numpy.typing import NDArray
import xrayutilities as xu
import copy
from xrayutilities.materials.spacegrouplattice import WyckoffBase
from enum import Enum
from typing import List, Self, Tuple, Dict


class Shape(Enum):
    """
    Diffraction peak shape selection for Gaussian, Lorentzian or neither peak shape.

    """

    Gaussian = 1
    Lorentzian = 2
    Neither = 3


class Sample:
    """
    A sample for powder diffraction containing different (solution) phases.

    Attributes:
    cifs (List[os.PathLike]): cif files each phase is calculated of
    concentration_sol (List[float]): concentration of solution phase components for each phase
    cryst_size List[float]: average crystal size in Angström for each phase
    vol_per_atom List[float]: volume per atom of each phase
    mol_amount List[float]: mol amount per phase in sample
    concentration_coex (np.ndarray): relative concentrations of each phase in the powder

    Functions:
    add_phase: add a phase to the powder
    set_composition: set the phase composition of the powder

    """

    def __init__(self):
        """
        Initialises a powder diffraction sample which can have multiple coexisting phases present.

        """
        self.cifs: List[NDArray] = []
        self.concentration_sol = []
        self.cryst_size = []
        self.vol_per_atom = []
        self.mol_amount = []
        self.concentration_coex = np.array([], dtype=object)

    def add_phase(self, cif: os.PathLike | str, mol_amount: float, cryst_size: float):
        """
        Adds a new phase to an existing sample.

        Args:
            cif (os.PathLike | str): cif file name of phase
            mol_amount (float): mol amount of phase in sample
            cryst_size (float): average crystal size in Angström for each phase

        """
        cif_path = os.path.join("cif", cif)
        if not Path(cif_path).is_file():
            raise ValueError(f"The CIF file ({cif_path}) does not exist.")

        # convert to Path object
        cif = Path(cif)

        self.cifs.append(np.array([cif], dtype=Path))
        self.concentration_sol.append(np.array([1.0]))  # at%
        self.cryst_size.append(cryst_size)  # meter
        self.mol_amount.append(mol_amount) # mol

    def set_composition(self):
        """
        Set the composition of the powder with concentrations of each of the coexisting phases.

        Raises:
            AssertionError:  number of concentrations has to match the number of phases
            AssertionError:  concentrations have to sum up to 1.0
        """
        tot_mol_amount = sum(self.mol_amount)
        for i in range(len(self.mol_amount)):
            self.concentration_coex[i] = self.mol_amount/tot_mol_amount

        assert len(self.concentration_coex) == len(
            self.cifs
        ), "The number of concentrations ({}) has to match the number of phases ({})".format(
            len(self.concentration_coex), len(self.cifs)
        )

        assert np.sum(self.concentration_coex) == 1.0, "The sum of all concentrations has to be 1.0"

    def calculate_diffractogram(self, sample, lambda_used=1.5406, two_theta=np.linspace(10, 135, 1000), shape=Shape.Gaussian):
        sample.set_composition()
        diff = Diffractogram(sample, lambda_used, two_theta, shape)
        diff.compute_intensity()
        return diff

    def __str__(self) -> str:
        flist = "\n".join(["    " + str(x[0]) for x in self.cifs])
        answer = [
            f"Sample {__class__}:",
            f"Number of phases: {len(self.cifs)}",
            f"Phase CIFs: ",
            f"{flist}",
            f"Concentration of coexisting phases: {self.concentration_coex}",
        ]
        return "\n".join(answer)


def create_sol_phase(
    cif_files_list: List[os.PathLike | str],
    concentration: List[float],
    output_file: os.PathLike | None = None,
) -> os.PathLike:
    """
    Creates a solution phase cif file from two seperate crystals based on concentration.

    Args:
        cif_files_list (List[os.PathLike]): cif files of input phases
        concentration (List[float]): concentration of input phases in solution phase
        save_generated_cif (bool, default=False): save the solution phase cif file

    Raises:
        ValueError: each input phase has to have a concentration value specified for
        ValueError: the concentrations of the input phases have to add up to 1.0

    Returns:
        (str): name of solution phase cif file saved in "cif" folder
    """

    # create array for phases in solution phase
    sol_phase = []
    cif_files = np.array(cif_files_list, dtype=str)
    concentration_sol = np.array(concentration)

    def combine_cif_names(input_files: List[os.PathLike | str]):
        """
        Combines the names of input cif files to name for solution phase cif file.

        Args:
            input_files (List[os.PathLike]): cif files of phases in solution phase

        Returns:
            (str): combined cif file name of solution phase
        """
        base_names = [os.path.splitext(os.path.basename(file))[0] for file in input_files]
        name_sol = "_".join(base_names) + "_sol.cif"
        return name_sol

    # check input
    if len(cif_files) != len(concentration_sol):
        raise ValueError("A concentration [at%] per end member has to be specified.")
    if np.sum(concentration_sol) != 1.0:  # i recommend the numpy.testing.assert_approx_equal approach from above
        raise ValueError("All solution phase concentrations have to sum up to 1.0.")

    # create material
    for cif_file_comp in cif_files:
        cif_path = os.path.join("cif", cif_file_comp)
        if not os.path.exists(cif_path):
            raise FileNotFoundError(f"The CIF file '{cif_file_comp}' does not exist.")

        sol_phase.append(xu.materials.Crystal.fromCIF(cif_path))

    # adapt lattice parameter
    # i am not sure deepcopy is needed anymore if you are using the new_sample
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

        # accessing the lattice._wbase indicates that this is supposed to be a private attribute
        # so this is likely not the best way to do this. Maybe the .base() function of the lattice provides something a little more robust?

        for atom, wyckoff, occ, b in sol_phase[j].lattice._wbase:
            new_occ = float(concentration_sol[j])  # Replace occupancy with concentration
            new_wbase.append(atom, wyckoff, occ=new_occ, b=b)

    # correct for 1.0
    material_sol.a += -1.0
    material_sol.b += -1.0
    material_sol.c += -1.0

    # Assign the new WyckoffBase to the lattice
    material_sol.lattice._wbase = new_wbase

    if output_file is None:
        name_sol = combine_cif_names(cif_files)
    else:
        name_sol = output_file

    # export to cif file
    material_sol.toCIF(os.path.join("cif", name_sol))

    return name_sol


class Diffractogram:
    """
    A diffractogram from a sample based on selected measuring parameters.

    Attributes:
    lamda_used (float): wavelength in Angström of diffractometer
    two_theta (np.ndarray): two_theta range
    shape (Shape): peak shape
    intensity List[float]: calculated intensity for each phase for two_theta range
    intensity (np.ndarray): calculated intensity for powder for two_theta range
    sample (Sample): sample diffractogram is calculated for

    Functions:
    compute_intensity: calculate intensity of diffractogram
    plot_diffractogram_matplotlib: plot intensity over two_theta using matplotlib

    """

    def __init__(self, sample: Sample, lambda_used: float, two_theta: np.ndarray, shape: Shape):
        """
        Initializes a diffractogram object

        Args:
            sample (Sample): sample to calculate powder diffractogram for
            lamda_used (float): wavelength in Angström of diffractometer
            two_theta (np.ndarray): two_theta range
            shape (Shape): peak shape

        Raises:
            ValueError: at least one phase needs to be present in the sample
            ValueError: the composition of coexisting phases has to be set in the sample

        """
        self.lambda_used = lambda_used
        self.two_theta = two_theta
        self.shape = shape
        self.intensity = []
        self.intensity_coex = np.zeros(len(self.two_theta))
        self.sample = sample

        if len(self.sample.cifs) == 0:
            raise ValueError("At least one layer has to be added to a sample in order to calculate a diffractogram.")
        if len(self.sample.concentration_coex) == 0:
            # I recommend changing this, as there is no clear indication (or function) to set 'phase composition'
            # in the sample class - or add that function
            raise ValueError("Set the phase composition first in order to calculate a diffractogram.")

    def compute_intensity(self):
        """
        Calculates the intensity of a powder diffractogram for a sample and saves intensity and volume per atom per phase.

        """
        for i, cif_files in enumerate(self.sample.cifs):
            inte_pdf, vol_pdf = cif_to_diffractogram(
                cif_file=cif_files.item(),
                lambda_used=self.lambda_used,
                shape=self.shape,
                cryst_size=self.sample.cryst_size[i],
                two_theta=self.two_theta,
            )
            self.intensity.append(inte_pdf)
            self.sample.vol_per_atom.append(vol_pdf)

        # calculate intensity for coexisting phases
        self.intensity_coex = combine_intensities(
            self.intensity, self.sample.vol_per_atom, self.sample.concentration_coex
        )

    def plot_diffractogram_matplotlib(self, **kwargs):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.two_theta, self.intensity_coex)
        plt.xlabel(kwargs.get("xlabel", "2θ/degrees"))
        plt.ylabel(kwargs.get("ylabel", "Intensity"))
        plt.gca().axes.yaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.title(kwargs.get("title", "Calculated Powder Diffraction Pattern"))
        plt.show(block=False)


def cif_to_diffractogram(
    cif_file: str,
    lambda_used=1.5406,
    shape=Shape.Gaussian,
    cryst_size=1e-7,
    two_theta=np.linspace(10, 135, 500),
):
    """
    Takes a cif file of a material and calculates the powder diffractogram from it.

    Args:
        cif_file (str): cif file name
        lambda_used (float, optional): wavelength in Angström of diffractometer, Defaults to 1.5406.
        shape (Shape, optional): peak shape, Defaults to Shape.Gaussian.
        cryst_size (float, optional): average crystallite size of phase, Defaults to 1e-7.
        two_theta (np.ndarray, optional): angular range for diffractogram, Defaults to np.linspace(10, 135, 500).

    Raises:
        FileNotFoundError: cif file has to be saved in "cif" folder

    Returns:
        I (np.ndarray): intensity of powder diffraction from input cif file
        V_at (float): volume per atom of this phase
    """
    # acknowledge processing
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


def combine_intensities(intensity: List[float], vol_per_atom: List[float], concentration_coex: np.ndarray):
    """
        Adds the intensities calculated for all coexisting phases based on the powder composition.

    Args:
        intensity (List[float]): intensity calculated for each phase
        vol_per_atom (List[float]): volume per atom calculated for each phase
        concentration_coex (np.ndarray): concentration of coexisting phases in powder

    Returns:
        _(np.ndarray): intensity of powder
    """
    vol_tot = np.sum(vol_per_atom * concentration_coex)

    # apply weight factor to intensity
    intensity_all = np.zeros(len(intensity[0][:]))
    for i, phase in enumerate(intensity):
        intensity_all += intensity[i][:] * (concentration_coex[i] * vol_per_atom[i] / vol_tot)

    return intensity_all


if __name__ == "__main__":

    # Example case 1
    sample_1 = Sample()
    sample_1.add_phase("Fe.cif", mol_amount=3, cryst_size=1e-7)
    new_cif_sol = create_sol_phase(["Fe.cif", "Ni.cif"], concentration=[0.3, 0.7])
    sample_1.add_phase(new_cif_sol, mol_amount=7, cryst_size=1e-7)
    diff_1 = sample_1.calculate_diffractogram(sample_1, lambda_used=1.5406, two_theta=np.linspace(10, 135, 1000), shape=Shape.Gaussian,)
    diff_1.plot_diffractogram_matplotlib()

    # Example 2
    sample_2 = Sample()
    sample_2.add_phase("Ni.cif", cryst_size=1e-7)
    sample_2.add_phase("Fe.cif", cryst_size=1e-7)
    sample_2.set_composition(concentration_coex=[0.1, 0.9])

    diff_2 = Diffractogram(
        sample_2,
        lambda_used=1.5406,
        two_theta=np.linspace(10, 135, 1000),
        shape=Shape.Gaussian,
    )
    diff_2.compute_intensity()
    diff_2.plot_diffractogram_matplotlib()
