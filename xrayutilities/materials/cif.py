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
# Copyright (C) 2010-2017 Dominik Kriegner <dominik.kriegner@gmail.com>

import itertools
import os
import re
import shlex
import warnings

import numpy

from . import elements
from .. import config
from .lattice import Lattice, LatticeBase
from .spacegrouplattice import SGLattice

re_loop = re.compile(r"^loop_")
re_symop = re.compile(r"^\s*("
                      "_space_group_symop_operation_xyz|"
                      "_symmetry_equiv_pos_as_xyz)")
re_name = re.compile(r"^\s*_chemical_formula_sum")
re_atom = re.compile(r"^\s*(_atom_site_label|_atom_site_type_symbol)\s*$")
re_atomx = re.compile(r"^\s*_atom_site_fract_x")
re_atomy = re.compile(r"^\s*_atom_site_fract_y")
re_atomz = re.compile(r"^\s*_atom_site_fract_z")
re_uiso = re.compile(r"^\s*_atom_site_U_iso_or_equiv")
re_atomocc = re.compile(r"^\s*_atom_site_occupancy")
re_labelline = re.compile(r"^\s*_")
re_emptyline = re.compile(r"^\s*$")
re_quote = re.compile(r"'")
re_cell_a = re.compile(r"^\s*_cell_length_a")
re_cell_b = re.compile(r"^\s*_cell_length_b")
re_cell_c = re.compile(r"^\s*_cell_length_c")
re_cell_alpha = re.compile(r"^\s*_cell_angle_alpha")
re_cell_beta = re.compile(r"^\s*_cell_angle_beta")
re_cell_gamma = re.compile(r"^\s*_cell_angle_gamma")
re_comment = re.compile(r"^\s*#")


class CIFFile(object):
    """
    class for parsing CIF (Crystallographic Information File) files. The class
    aims to provide an additional way of creating material classes instead of
    manual entering of the information the lattice constants and unit cell
    structure are parsed from the CIF file
    """

    def __init__(self, filename, digits=3):
        """
        initialization of the CIFFile class

        Parameters
        ----------
         filename:  filename of the CIF file
         digits:    number of digits to check if position is unique (optional)
        """
        self.name = os.path.splitext(os.path.split(filename)[-1])[0]
        self.filename = filename
        self.digits = digits

        try:
            self.fid = open(self.filename, "rb")
        except:
            raise IOError("cannot open CIF file %s" % self.filename)

        self.Parse()
        self.SymStruct()

    def __del__(self):
        """
        class destructor which closes open files
        """
        if self.fid is not None:
            self.fid.close()

    def Parse(self):
        """
        function to parse a CIF file. The function reads the
        space group symmetry operations and the basic atom positions
        as well as the lattice constants and unit cell angles
        """

        self.symops = []
        self.atoms = []
        self.lattice_const = numpy.zeros(3, dtype=numpy.double)
        self.lattice_angles = numpy.zeros(3, dtype=numpy.double)

        self.fid.seek(0)  # set file pointer to the beginning
        loop_start = False
        symop_loop = False
        atom_loop = False

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

        for line in self.fid.readlines():
            line = line.decode('ascii', 'ignore')
            if config.VERBOSITY >= config.DEBUG:
                print(line)

            # ignore comment lines
            if re_comment.match(line):
                continue

            if re_loop.match(line):  # start of loop
                if config.VERBOSITY >= config.DEBUG:
                    print('XU.material: loop start found')
                loop_start = True
                loop_labels = []
                symop_loop = False
                atom_loop = False
                ax_idx = None
                ay_idx = None
                az_idx = None
                uiso_idx = None
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
                elif re_name.match(line):
                    try:
                        self.name = shlex.split(line)[1]
                    except:
                        pass
                if loop_start:
                    loop_labels.append(line.strip())
                    if re_symop.match(line):  # start of symmetry op. loop
                        if config.VERBOSITY >= config.DEBUG:
                            print('XU.material: symop-loop identified')
                        symop_loop = True
                        symop_idx = len(loop_labels) - 1
                    elif re_atom.match(line):  # start of atom position loop
                        if config.VERBOSITY >= config.DEBUG:
                            print('XU.material: atom position-loop identified')
                        atom_loop = True
                        alab_idx = len(loop_labels) - 1
                    elif re_atomx.match(line):
                        ax_idx = len(loop_labels) - 1
                        if config.VERBOSITY >= config.DEBUG:
                            print('XU.material: atom position x: col%d'
                                  % ax_idx)
                    elif re_atomy.match(line):
                        ay_idx = len(loop_labels) - 1
                    elif re_atomz.match(line):
                        az_idx = len(loop_labels) - 1
                    elif re_uiso.match(line):
                        uiso_idx = len(loop_labels) - 1
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
                # add a comma to a fraction to avoid int division problems
                opstr = re.sub(r"/([1-9])", r"/\1.", opstr)
                self.symops.append(opstr)
            elif atom_loop:  # atom label and position
                loop_start = False
                asplit = line.split()
                alabel = asplit[alab_idx]
                apos = (floatconv(asplit[ax_idx]),
                        floatconv(asplit[ay_idx]),
                        floatconv(asplit[az_idx]))
                occ = floatconv(asplit[occ_idx]) if occ_idx else 1
                uiso = floatconv(asplit[uiso_idx]) if uiso_idx else 0
                biso = 8 * numpy.pi**2 * uiso
                self.atoms.append((alabel, apos, occ, biso))

    def SymStruct(self):
        """
        function to obtain the list of different atom positions
        in the unit cell for the different types of atoms. The data
        are obtained from the data parsed from the CIF file.
        """

        self.unique_positions = []
        for a in self.atoms:
            unique_pos = []
            x = a[1][0]
            y = a[1][1]
            z = a[1][2]
            el = re.sub(r"['\"]", r"", a[0])
            el = re.sub(r"([0-9])", r"", el)
            el = re.sub(r"\(\w*\)", r"", el)
            for symop in self.symops:
                pos = eval("numpy.array(" + symop + ")")
                # check that position is within unit cell
                pos = pos - numpy.round(pos, self.digits) // 1
                # check if position is unique
                unique = True
                for upos in unique_pos:
                    if (numpy.round(upos, self.digits) ==
                            numpy.round(pos, self.digits)).all():
                        unique = False
                if unique:
                    unique_pos.append(pos)
            element = getattr(elements, el)
            self.unique_positions.append((element, unique_pos, a[2], a[3]))

    def Lattice(self):
        """
        returns a lattice object with the structure from the CIF file
        """
        warnings.warn("deprecated function -> change to SGLattice",
                      DeprecationWarning)
        lb = LatticeBase()
        for element, positions, occ, biso in self.unique_positions:
            for pos in positions:
                lb.append(element, pos, occ=occ, b=biso)

        # unit cell vectors
        ca = numpy.cos(numpy.radians(self.lattice_angles[0]))
        cb = numpy.cos(numpy.radians(self.lattice_angles[1]))
        cg = numpy.cos(numpy.radians(self.lattice_angles[2]))
        sa = numpy.sin(numpy.radians(self.lattice_angles[0]))
        sb = numpy.sin(numpy.radians(self.lattice_angles[1]))
        sg = numpy.sin(numpy.radians(self.lattice_angles[2]))

        a1 = self.lattice_const[0] * numpy.array([1, 0, 0], dtype=numpy.double)
        a2 = self.lattice_const[1] * \
            numpy.array([cg, sg, 0], dtype=numpy.double)
        a3 = self.lattice_const[2] * numpy.array([
            cb,
            (ca - cb * cg) / sg,
            numpy.sqrt(1 - ca ** 2 - cb ** 2 - cg ** 2 +
                       2 * ca * cb * cg) / sg],
            dtype=numpy.double)
        # create lattice
        l = Lattice(a1, a2, a3, base=lb)

        return l

    def SGLattice(self):
        """
        create a SGLattice object with the structure from the CIF file
        """
        atoms = []
        pos = []
        occ = []
        biso = []
        for element, positions, o, b in self.unique_positions:
            for p in positions:
                atoms.append(element)
                pos.append(('1a', p))
                occ.append(o)
                biso.append(b)

        return SGLattice(1, *itertools.chain(self.lattice_const,
                                             self.lattice_angles),
                         atoms=atoms, pos=pos, occ=occ, b=biso)

    def __str__(self):
        """
        returns a string with positions and names of the atoms
        """
        ostr = ""
        ostr += "unit cell structure\n"
        ostr += "a: %8.4f b: %8.4f c: %8.4f\n" % tuple(self.lattice_const)
        ostr += "alpha: %6.2f beta: %6.2f gamma: %6.2f\n" % tuple(
            self.lattice_angles)
        ostr += "Unique atom positions in unit cell\n"
        for atom in self.unique_positions:
            ostr += atom[0].name + " (%d): \n" % atom[0].num
            for pos in atom[1]:
                ostr += str(numpy.round(pos, self.digits)) + "\n"
        return ostr
