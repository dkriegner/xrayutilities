# This file is part of xrayutilities.
#
# xrayutilities is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (c) 2010-2021, 2023 Dominik Kriegner <dominik.kriegner@gmail.com>

import copy
import io
import itertools
import operator
import os
import re
import shlex

import numpy

from .. import config
from . import elements
from . import spacegrouplattice as sgl
from . import wyckpos

re_data = re.compile(r"^data_", re.IGNORECASE)
re_loop = re.compile(r"^loop_", re.IGNORECASE)
re_symop = re.compile(
    r"^\s*("
    "_space_group_symop_operation_xyz|"
    "_symmetry_equiv_pos_as_xyz)",
    re.IGNORECASE,
)
re_name = re.compile(r"^\s*_chemical_formula_sum", re.IGNORECASE)
re_atom = re.compile(r"^\s*_atom_site_label\s*$", re.IGNORECASE)
re_atomtyp = re.compile(r"^\s*_atom_site_type_symbol\s*$", re.IGNORECASE)
re_atomx = re.compile(r"^\s*_atom_site_fract_x", re.IGNORECASE)
re_atomy = re.compile(r"^\s*_atom_site_fract_y", re.IGNORECASE)
re_atomz = re.compile(r"^\s*_atom_site_fract_z", re.IGNORECASE)
re_uiso = re.compile(r"^\s*_atom_site_U_iso_or_equiv", re.IGNORECASE)
re_biso = re.compile(r"^\s*_atom_site_B_iso_or_equiv", re.IGNORECASE)
re_atomocc = re.compile(r"^\s*_atom_site_occupancy", re.IGNORECASE)
re_labelline = re.compile(r"^\s*_")
re_emptyline = re.compile(r"^\s*$")
re_quote = re.compile(r"'")
re_spacegroupnr = re.compile(
    r"^\s*(_space_group_IT_number|"
    "_symmetry_Int_Tables_number)",
    re.IGNORECASE,
)
re_spacegroupname = re.compile(
    r"^\s*(_symmetry_space_group_name_H-M|"
    "_space_group_name_H-M_alt)",
    re.IGNORECASE,
)
re_spacegroupsetting = re.compile(r"^\s*_symmetry_cell_setting", re.IGNORECASE)
re_cell_a = re.compile(r"^\s*_cell_length_a", re.IGNORECASE)
re_cell_b = re.compile(r"^\s*_cell_length_b", re.IGNORECASE)
re_cell_c = re.compile(r"^\s*_cell_length_c", re.IGNORECASE)
re_cell_alpha = re.compile(r"^\s*_cell_angle_alpha", re.IGNORECASE)
re_cell_beta = re.compile(r"^\s*_cell_angle_beta", re.IGNORECASE)
re_cell_gamma = re.compile(r"^\s*_cell_angle_gamma", re.IGNORECASE)
re_comment = re.compile(r"^\s*#")


class CIFFile:
    """
    class for parsing CIF (Crystallographic Information File) files. The class
    aims to provide an additional way of creating material classes instead of
    manual entering of the information the lattice constants and unit cell
    structure are parsed from the CIF file.

    If multiple datasets are present in the CIF file this class will attempt to
    parse all of them into the the data dictionary. By default all methods
    access the first data set found in the file.
    """

    def __init__(self, filestr, digits=4):
        """
        initialization of the CIFFile class

        Parameters
        ----------
        filestr :  str, bytes
            CIF filename or string representation of the CIF file
        digits :    int, optional
            number of digits to check if position is unique
        """
        self.digits = digits

        if os.path.isfile(filestr):
            self.filename = filestr
            try:
                fid = open(self.filename, "rb")
            except OSError as exc:
                raise IOError(f"cannot open CIF file {self.filename}") from exc
        else:
            if filestr.count("\n") == 0:
                print(
                    'XU.materials.CIFFile: "filestr" contains only one line '
                    "but a file with that name does not exist! Continuing "
                    "with the assumption this one line string is the "
                    "content of a CIF file!"
                )
            self.filename = "__from_str__"
            if isinstance(filestr, bytes):
                fid = io.BytesIO(filestr)
            else:
                fid = io.BytesIO(bytes(filestr.encode("ascii")))

        if config.VERBOSITY >= config.INFO_ALL:
            print(f"XU.materials: parsing CIF file {self.filename}")
        self.default_dataset = None
        self.data = {}
        self.Parse(fid)
        fid.close()

    def Parse(self, fid):
        """
        function to parse a CIF file. The function reads all the included data
        sets and adds them to the data dictionary.

        """
        fidpos = fid.tell()
        while True:
            line = fid.readline()
            if not line:
                break
            fidpos = fid.tell()
            line = line.decode("ascii", "ignore")
            m = re_data.match(line)
            if m:
                fid.seek(fidpos)
                name = line[m.end() :].strip()
                self.data[name] = CIFDataset(fid, name, self.digits)
                if self.data[name].has_atoms and not self.default_dataset:
                    self.default_dataset = name

    def SGLattice(self, dataset=None, use_p1=False):
        """
        create a SGLattice object with the structure from the CIF dataset

        Parameters
        ----------
        dataset :   str, optional
            name of the dataset to use. if None the default one will be used.
        use_p1 :    bool, optional
            force the use of P1 symmetry, default False
        """
        if not dataset:
            dataset = self.default_dataset
        return self.data[dataset].SGLattice(use_p1=use_p1)

    def __str__(self):
        """
        returns a string with positions and names of the atoms for all datasets
        """
        ostr = ""
        ostr += f"CIF-File: {self.filename}\n"
        for name, dataset in self.data.items():
            ostr += f"\nDataset: {name}"
            if name == self.default_dataset:
                ostr += " (default)"
            ostr += "\n"
            ostr += str(dataset)
        return ostr


class CIFDataset:
    """
    class for parsing CIF (Crystallographic Information File) files. The class
    aims to provide an additional way of creating material classes instead of
    manual entering of the information the lattice constants and unit cell
    structure are parsed from the CIF file
    """

    def __init__(self, fid, name, digits):
        """
        initialization of the CIFDataset class. This class parses one data
        block.

        Parameters
        ----------
        fid :       filehandle
            file handle set to the beginning of the data block to be parsed
        name :      str
            identifier string of the dataset
        digits :    int
            number of digits to check if position is unique
        """
        self.name = name
        self.digits = digits
        self.has_atoms = False

        if config.VERBOSITY >= config.INFO_ALL:
            print(f"XU.materials: parsing cif dataset {self.name}")
        self.Parse(fid)
        self.SymStruct()

    def Parse(self, fid):
        """
        function to parse a CIF data set. The function reads the
        space group symmetry operations and the basic atom positions
        as well as the lattice constants and unit cell angles
        """

        self.symops = []
        self.atoms = []
        self.lattice_const = numpy.zeros(3, dtype=numpy.double)
        self.lattice_angles = numpy.zeros(3, dtype=numpy.double)

        loop_start = False
        symop_loop = False
        atom_loop = False

        def get_element(cifstring):
            el = re.sub(r"['\"]", r"", cifstring)
            if "+" in el or "-" in el:
                # add oxidation number if not present
                for sign in ("+", "-"):
                    if sign in el:
                        if not el[el.index(sign) - 1].isdigit():
                            signidx = el.index(sign)
                            el = el[:signidx] + "1" + el[signidx:]
                # replace special characters
                for r, o in zip(("dot", "p", "m"), (".", "+", "-")):
                    el = el.replace(o, r)
            else:
                el = re.sub(r"([0-9])", r"", el)
            el = re.sub(r"\(\w*\)", r"", el)
            try:
                element = getattr(elements, el)
            except AttributeError:  # el not found, typ. due to oxidation state
                f = re.search("[0-9]", el)
                if not f and el == "?":
                    element = elements.Dummy
                elif f is None:
                    raise ValueError(
                        "XU.materials: element ('%s') could not"
                        " be identified as chemical element. Only"
                        " abbreviations of element names are "
                        "supported." % (cifstring)
                    )
                else:
                    elname = el[: f.start()]
                    if hasattr(elements, elname):
                        # here one might want to find a closer alternative than
                        # the neutral atom, but the effect this has should be
                        # minimal, currently simply the neutral atom is used
                        if config.VERBOSITY >= config.INFO_LOW:
                            print(
                                f"XU.materials: Warning: element {elname} "
                                f"used instead of {cifstring}"
                            )
                        element = getattr(elements, elname)
                    else:
                        raise ValueError(
                            "XU.materials: element ('%s') could "
                            "not be found" % (cifstring)
                        )
            return element

        def floatconv(string):
            """
            helper function to convert string with possible error
            given in brackets to float
            """
            try:
                f = float(re.sub(r"\(.+\)", r"", string))
            except ValueError:
                f = numpy.nan
            return f

        def intconv(string):
            """
            helper function to convert string to integer
            """
            try:
                i = int(string)
            except ValueError:
                i = None
            return i

        fidpos = fid.tell()
        for line in fid.readlines():
            linelen = len(line)
            line = line.decode("ascii", "ignore")
            if config.VERBOSITY >= config.DEBUG:
                print(line)
                print(fid.tell(), fidpos)

            if re_data.match(line):
                fid.seek(fidpos)
                break
            fidpos += linelen

            # ignore comment lines
            if re_comment.match(line):
                continue

            if re_loop.match(line):  # start of loop
                if config.VERBOSITY >= config.DEBUG:
                    print("XU.materials: loop start found")
                loop_start = True
                loop_labels = []
                symop_loop = False
                symop_idx = None
                atom_loop = False
                alab_idx = None
                ax_idx = None
                ay_idx = None
                az_idx = None
                uiso_idx = None
                biso_idx = None
                occ_idx = None
            elif re_labelline.match(line):
                if re_cell_a.match(line):
                    self.lattice_const[0] = floatconv(line.split()[1])
                elif re_cell_b.match(line):
                    self.lattice_const[1] = floatconv(line.split()[1])
                elif re_cell_c.match(line):
                    self.lattice_const[2] = floatconv(line.split()[1])
                elif re_cell_alpha.match(line):
                    self.lattice_angles[0] = floatconv(line.split()[1])
                elif re_cell_beta.match(line):
                    self.lattice_angles[1] = floatconv(line.split()[1])
                elif re_cell_gamma.match(line):
                    self.lattice_angles[2] = floatconv(line.split()[1])
                elif re_spacegroupnr.match(line):
                    i = intconv(line.split()[1])
                    if i:
                        self.sgrp_nr = i
                elif re_spacegroupname.match(line):
                    self.sgrp_name = "".join(line.split()[1:]).strip("'")
                elif re_spacegroupsetting.match(line):
                    i = intconv(line.split()[1])
                    if i:
                        self.sgrp_setting = i
                elif re_name.match(line):
                    try:
                        self.name = shlex.split(line)[1]
                    except IndexError:
                        self.name = None
                if loop_start:
                    loop_labels.append(line.strip())
                    if re_symop.match(line):  # start of symmetry op. loop
                        if config.VERBOSITY >= config.DEBUG:
                            print("XU.materials: symop-loop identified")
                        symop_loop = True
                        symop_idx = len(loop_labels) - 1
                    elif re_atom.match(line) or re_atomtyp.match(line):
                        # start of atom position loop
                        if config.VERBOSITY >= config.DEBUG:
                            print("XU.materials: atom position-loop found")
                        atom_loop = True
                        if re_atomtyp.match(line):
                            alab_idx = len(loop_labels) - 1
                        elif not list(filter(re_atomtyp.match, loop_labels)):
                            # ensure precedence of atom_site_type_symbol
                            alab_idx = len(loop_labels) - 1
                    elif re_atomx.match(line):
                        ax_idx = len(loop_labels) - 1
                        if config.VERBOSITY >= config.DEBUG:
                            print(
                                "XU.materials: atom position x: col%d" % ax_idx
                            )
                    elif re_atomy.match(line):
                        ay_idx = len(loop_labels) - 1
                    elif re_atomz.match(line):
                        az_idx = len(loop_labels) - 1
                    elif re_uiso.match(line):
                        uiso_idx = len(loop_labels) - 1
                    elif re_biso.match(line):
                        biso_idx = len(loop_labels) - 1
                    elif re_atomocc.match(line):
                        occ_idx = len(loop_labels) - 1

            elif re_emptyline.match(line):
                loop_start = False
                symop_loop = False
                atom_loop = False
                continue
            elif symop_loop:  # symmetry operation entry
                loop_start = False
                entry = shlex.split(line)[symop_idx]
                if re_quote.match(entry):
                    opstr = entry
                else:
                    opstr = "'" + entry + "'"
                opstr = re.sub(r"^'", r"(", opstr)
                opstr = re.sub(r"'$", r")", opstr)
                self.symops.append(opstr)
            elif atom_loop:  # atom label and position
                loop_start = False
                asplit = line.split()
                try:
                    atom = get_element(asplit[alab_idx])
                    apos = (
                        floatconv(asplit[ax_idx]),
                        floatconv(asplit[ay_idx]),
                        floatconv(asplit[az_idx]),
                    )
                    occ = floatconv(asplit[occ_idx]) if occ_idx else 1
                    if numpy.isnan(occ):
                        occ = 1
                    uiso = floatconv(asplit[uiso_idx]) if uiso_idx else 0
                    biso = floatconv(asplit[biso_idx]) if biso_idx else 0
                    if numpy.isnan(uiso):
                        uiso = 0
                    if numpy.isnan(biso):
                        biso = 0
                    if biso == 0:
                        biso = 8 * numpy.pi**2 * uiso
                    self.atoms.append((atom, apos, occ, biso))
                except IndexError:
                    if config.VERBOSITY >= config.INFO_LOW:
                        print(
                            'XU.materials: could not parse atom line: "%s"'
                            % line.strip()
                        )
        if self.atoms:
            self.has_atoms = True

    def SymStruct(self):
        """
        function to obtain the list of different atom positions in the unit
        cell for the different types of atoms and determine the space group
        number and origin choice if available. The data are obtained from the
        data parsed from the CIF file.
        """

        def rem_white(string):
            return string.replace(" ", "")

        if hasattr(self, "sgrp_name"):
            # determine spacegroup
            cifsgn = rem_white(self.sgrp_name).split(":")[0]
            for nr, name in sgl.sgrp_name.items():
                if cifsgn == rem_white(name):
                    self.sgrp_nr = int(nr)
            if not hasattr(self, "sgrp_nr"):
                # try ignoring the minuses
                for nr, name in sgl.sgrp_name.items():
                    if cifsgn == rem_white(name.replace("-", "")):
                        self.sgrp_nr = int(nr)
            if len(self.sgrp_name.split(":")) > 1:
                self.sgrp_suf = ":" + self.sgrp_name.split(":")[1]
            elif hasattr(self, "sgrp_setting"):
                self.sgrp_suf = ":%d" % self.sgrp_setting

        if not hasattr(self, "sgrp_suf"):
            if hasattr(self, "sgrp_nr"):
                self.sgrp_suf = sgl.get_possible_sgrp_suf(self.sgrp_nr)
            else:
                self.sgrp_suf = ""
        if isinstance(self.sgrp_suf, str):
            suffixes = [
                self.sgrp_suf,
            ]
        else:
            suffixes = copy.copy(self.sgrp_suf)
        for sgrp_suf in suffixes:
            self.sgrp_suf = sgrp_suf
            if hasattr(self, "sgrp_nr"):
                self.sgrp = str(self.sgrp_nr) + self.sgrp_suf
                if config.VERBOSITY >= config.INFO_ALL:
                    print(f"XU.materials: trying space group {self.sgrp}")

            # determine all unique positions for definition of a P1 space group
            symops = self.symops
            if not symops and hasattr(self, "sgrp") and self.atoms:
                label = sorted(
                    wyckpos.wp[self.sgrp], key=lambda s: int(s[:-1])
                )
                symops = wyckpos.wp[self.sgrp][label[-1]][1]
                if config.VERBOSITY >= config.INFO_ALL:
                    print(
                        "XU.materials: no symmetry operations in CIF-Dataset"
                        "; using built in general positions."
                    )
            self.unique_positions = []
            for el, (x, y, z), occ, biso in self.atoms:
                unique_pos = set()
                for symop in symops:
                    pos = eval(symop, {"x": x, "y": y, "z": z})
                    pos = numpy.asarray(pos)
                    # check that position is within unit cell
                    pos = pos - numpy.round(pos, self.digits) // 1
                    unique_pos.add(tuple(numpy.round(pos, self.digits)))
                self.unique_positions.append((el, unique_pos, occ, biso))

            # determine Wyckoff positions and free parameters of unit cell
            if hasattr(self, "sgrp"):
                self.wp = []
                self.occ = []
                self.elements = []
                self.biso = []
                keys = list(wyckpos.wp[self.sgrp])
                wpn = [int(re.sub(r"([a-zA-Z])", r"", k)) for k in keys]
                for i, (el, (x, y, z), occ, biso) in enumerate(self.atoms):
                    # candidate positions from number of unique atoms
                    natoms = len(self.unique_positions[i][1])
                    wpcand = []
                    for j, n in enumerate(wpn):
                        if n == natoms:
                            wpcand.append(
                                (keys[j], wyckpos.wp[self.sgrp][keys[j]])
                            )
                    for j, (k, wp) in enumerate(
                        sorted(wpcand, key=operator.itemgetter(1))
                    ):
                        parint, poslist, _ = wp
                        for positem in poslist:
                            foundwp, xyz = sgl.testwp(
                                parint, positem, (x, y, z), self.digits
                            )
                            if foundwp:
                                if xyz is None:
                                    self.wp.append(k)
                                else:
                                    self.wp.append((k, xyz))
                                self.elements.append(el)
                                self.occ.append(occ)
                                self.biso.append(biso)
                                break
                        if foundwp:
                            break
                if config.VERBOSITY >= config.INFO_ALL:
                    print(
                        f"XU.materials: {len(self.wp):d} of "
                        f"{len(self.atoms):d} Wyckoff positions identified"
                    )
                    if len(self.wp) < len(self.atoms):
                        print(
                            f"XU.materials: space group {self.sgrp} seems "
                            "not to fit"
                        )

                # free unit cell parameters
                self.crystal_system, _ = sgl.sgrp_sym[self.sgrp_nr]
                self.crystal_system += self.sgrp_suf
                self.uc_params = []
                p2i = {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                    "alpha": 0,
                    "beta": 1,
                    "gamma": 2,
                }
                for pname in sgl.sgrp_params[self.crystal_system][0]:
                    if pname in ("a", "b", "c"):
                        self.uc_params.append(self.lattice_const[p2i[pname]])
                    if pname in ("alpha", "beta", "gamma"):
                        self.uc_params.append(self.lattice_angles[p2i[pname]])

                if len(self.wp) == len(self.atoms):
                    if config.VERBOSITY >= config.INFO_ALL:
                        print(
                            "XU.materials: identified space group as "
                            f"{self.sgrp}"
                        )
                    break

    def SGLattice(self, use_p1=False):
        """
        create a SGLattice object with the structure from the CIF file
        """
        if not use_p1:
            if hasattr(self, "sgrp"):
                if len(self.wp) == len(self.atoms):
                    return sgl.SGLattice(
                        self.sgrp,
                        *self.uc_params,
                        atoms=self.elements,
                        pos=self.wp,
                        occ=self.occ,
                        b=self.biso,
                    )
                if config.VERBOSITY >= config.INFO_LOW:
                    print("XU.materials: Wyckoff positions missing, using P1")
            else:
                if config.VERBOSITY >= config.INFO_LOW:
                    print(
                        "XU.materials: space-group detection failed, using P1"
                    )

        atomdict = {"atoms": [], "pos": [], "occ": [], "b": []}
        for element, positions, o, b in self.unique_positions:
            for p in positions:
                atomdict["atoms"].append(element)
                atomdict["pos"].append(("1a", p))
                atomdict["occ"].append(o)
                atomdict["b"].append(b)

        return sgl.SGLattice(
            1,
            *itertools.chain(self.lattice_const, self.lattice_angles),
            **atomdict,
        )

    def __str__(self):
        """
        returns a string with positions and names of the atoms
        """
        ostr = ""
        ostr += "unit cell structure:"
        if hasattr(self, "sgrp"):
            ostr += " %s %s %s\n" % (
                self.sgrp,
                self.crystal_system,
                getattr(self, "sgrp_name", ""),
            )
        else:
            ostr += "\n"
        ostr += "a: %8.4f b: %8.4f c: %8.4f\n" % tuple(self.lattice_const)
        ostr += "alpha: %6.2f beta: %6.2f gamma: %6.2f\n" % tuple(
            self.lattice_angles
        )
        if self.unique_positions:
            ostr += "Unique atom positions in unit cell\n"
        for atom in self.unique_positions:
            ostr += atom[0].name + " (%d): \n" % atom[0].num
            for pos in atom[1]:
                ostr += str(numpy.round(pos, self.digits)) + "\n"
        return ostr


def cifexport(filename, mat):
    """
    function to export a Crystal instance to CIF file. This in particular
    includes the atomic coordinates, however, ignores for example the elastic
    parameters.
    """

    def unique_label(basename, names):
        num = 1
        name = f"{basename}{num:d}"
        while name in names:
            num += 1
            name = f"{basename}{num:d}"
        return name

    general = """data_global
_chemical_formula_sum '{chemsum}'
_cell_length_a {a:.5f}
_cell_length_b {b:.5f}
_cell_length_c {c:.5f}
_cell_angle_alpha {alpha:.4f}
_cell_angle_beta {beta:.4f}
_cell_angle_gamma {gamma:.4f}
_cell_volume {vol:.3f}
_space_group_crystal_system {csystem}
_space_group_IT_number {sgrpnr}
_space_group_name_H-M_alt '{hmsymb}'
"""

    csystem = mat.lattice.crystal_system
    if len(mat.lattice.space_group_suf) > 0:
        csystem = csystem[: -len(mat.lattice.space_group_suf)]

    ctxt = general.format(
        chemsum=mat.chemical_composition(with_spaces=True),
        a=mat.a,
        b=mat.b,
        c=mat.c,
        alpha=mat.alpha,
        beta=mat.beta,
        gamma=mat.gamma,
        vol=mat.lattice.UnitCellVolume(),
        csystem=csystem,
        sgrpnr=mat.lattice.space_group_nr,
        hmsymb=mat.lattice.name,
    )

    sgrpsuf = mat.lattice.space_group_suf[1:]
    if sgrpsuf:
        ctxt += f"_symmetry_cell_setting {sgrpsuf}\n"

    symloop = """
loop_
_space_group_symop_operation_xyz
"""

    for symop in mat.lattice.symops:
        symloop += "'" + symop.xyz() + "'\n"

    atomloop = """
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_symmetry_multiplicity
_atom_site_Wyckoff_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_B_iso_or_equiv
"""
    nidx = 0
    allatoms = list(mat.lattice.base())
    names = []
    for at, pos, occ, b in mat.lattice._wbase:
        wm, wl, _ = re.split("([a-z])", pos[0])
        nsite = int(wm)
        x, y, z = allatoms[nidx][1]
        names.append(unique_label(at.name, names))
        atomloop += "%s %s %d %c %.5f %.5f %.5f %.4f %.4f\n" % (
            names[-1],
            at.name,
            nsite,
            wl,
            x,
            y,
            z,
            occ,
            b,
        )
        nidx += nsite

    with open(filename, "w") as f:
        f.write(ctxt)
        f.write(symloop)
        f.write(atomloop)
