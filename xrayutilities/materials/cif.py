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
from __future__ import division

import itertools
import operator
import os
import re
import shlex
import warnings

import numpy
import scipy.optimize

from . import spacegrouplattice as sgl
from . import elements, wyckpos
from .. import config
from .lattice import Lattice, LatticeBase

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
re_spacegroupnr = re.compile(r"^\s*_space_group_IT_number|"
                             "_symmetry_Int_Tables_number")
re_spacegroupname = re.compile(r"^\s*_symmetry_space_group_name_H-M")
re_spacegroupsetting = re.compile(r"^\s*_symmetry_cell_setting")
re_cell_a = re.compile(r"^\s*_cell_length_a")
re_cell_b = re.compile(r"^\s*_cell_length_b")
re_cell_c = re.compile(r"^\s*_cell_length_c")
re_cell_alpha = re.compile(r"^\s*_cell_angle_alpha")
re_cell_beta = re.compile(r"^\s*_cell_angle_beta")
re_cell_gamma = re.compile(r"^\s*_cell_angle_gamma")
re_comment = re.compile(r"^\s*#")


def testwp(parint, wp, cifpos, digits):
    """
    test if a Wyckoff position can describe the given position from a CIF file

    Parameters
    ----------
     parint:   integer telling which Parameters the given Wyckoff position has
     wp:       expression of the Wyckoff position (string of tuple)
     cifpos:   (x,y,z) position of the atom in the CIF file
     digits:   number of digits for which for a comparison of floating point
               numbers will be rounded to

    Returns
    -------
    foundflag, pars:  flag to tell if the positions match and if necessary any
                      parameters associated with the position
    """
    def check_numbers_match(p1, p2, digits):
        p1 = p1 - numpy.round(p1, digits) // 1
        p2 = p2 - numpy.round(p2, digits) // 1
        if numpy.round(p1, digits) == numpy.round(p2, digits):
            return True
        else:
            return False

    def get_pardict(parint, x):
        i = 0
        pardict = {}
        if parint & 1:
            pardict['x'] = x[i]
            i += 1
        if parint & 2:
            pardict['y'] = x[i]
            i += 1
        if parint & 4:
            pardict['z'] = x[i]
        return pardict

    wyckp = wp.strip('()').split(',')
    # test agreement in positions witout variables
    match = numpy.asarray([False, False, False])
    variables = []
    for i in range(3):
        v = re.findall(r'[xyz]', wyckp[i])
        if v == []:
            pos = eval(wyckp[i])
            match[i] = check_numbers_match(pos, cifpos[i], digits)
            if not match[i]:
                return False, None
        else:
            variables.append(*v)

    if numpy.all(match):
        return True, None

    # check if with proper choice of the variables a correspondence of the
    # positions can be obtained
    def fmin(x, parint, wyckp, cifpos):
        evalexp = []
        cifp = []
        for i in range(3):
            if not match[i]:
                evalexp.append(wyckp[i])
                cifp.append(cifpos[i])
        pardict = get_pardict(parint, x)
        wpos = [eval(e, pardict) for e in evalexp]
        return numpy.linalg.norm(numpy.asarray(wpos)-numpy.asarray(cifp))

    x0 = []
    if 'x' in variables:
        x0.append(cifpos[0])
    if 'y' in variables:
        x0.append(cifpos[1])
    if 'z' in variables:
        x0.append(cifpos[2])

    opt = scipy.optimize.minimize(fmin, x0, args=(parint, wyckp, cifpos))
    pardict = get_pardict(parint, opt.x)
    for i in range(3):
        if not match[i]:
            pos = eval(wyckp[i], pardict)
            match[i] = check_numbers_match(pos, cifpos[i], digits)
    if numpy.all(match):
        return True, opt.x
    else:
        return False, None


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

        if config.VERBOSITY >= config.INFO_ALL:
            print('XU.material: parsing cif file %s' % self.filename)
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
                elif re_spacegroupnr.match(line):
                    self.sgrp_nr = int(line.split()[1])
                elif re_spacegroupname.match(line):
                    self.sgrp_name = ''.join(line.split()[1:]).strip("'")
                elif re_spacegroupsetting.match(line):
                    try:
                        self.sgrp_setting = int(line.split()[1])
                    except:
                        pass
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
        function to obtain the list of different atom positions in the unit
        cell for the different types of atoms and determine the space group
        number and origin choice if available. The data are obtained from the
        data parsed from the CIF file.
        """

        def rem_white(string):
            return string.replace(' ', '')

        if hasattr(self, 'sgrp_name'):
            # determine spacegroup
            cifsgn = rem_white(self.sgrp_name).split(':')[0]
            for nr, name in sgl.sgrp_name.items():
                if cifsgn == rem_white(name):
                    self.sgrp_nr = int(nr)
            if not hasattr(self, 'sgrp_nr'):
                # try ignoring the minuses
                for nr, name in sgl.sgrp_name.items():
                    if cifsgn == rem_white(name.replace('-', '')):
                        self.sgrp_nr = int(nr)
            if len(self.sgrp_name.split(':')) > 1:
                self.sgrp_suf = ':' + self.sgrp_name.split(':')[1]
            elif hasattr(self, 'sgrp_setting'):
                self.sgrp_suf = ':%d' % self.sgrp_setting

        if not hasattr(self, 'sgrp_suf'):
            if hasattr(self, 'sgrp_nr'):
                self.sgrp_suf = sgl.get_default_sgrp_suf(self.sgrp_nr)
        if hasattr(self, 'sgrp_nr'):
            self.sgrp = str(self.sgrp_nr) + self.sgrp_suf
            if config.VERBOSITY >= config.INFO_ALL:
                print('XU.material: space group identified as %s' % self.sgrp)

        # determine all unique positions for definition of a P1 space group
        def get_element_name(cifstring):
            el = re.sub(r"['\"]", r"", cifstring)
            el = re.sub(r"([0-9])", r"", el)
            el = re.sub(r"\(\w*\)", r"", el)
            return el

        self.unique_positions = []
        for cifel, (x, y, z), occ, biso in self.atoms:
            unique_pos = []
            el = get_element_name(cifel)
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
            self.unique_positions.append((element, unique_pos, occ, biso))

        # determine Wyckoff positions and free parameters of unit cell
        if hasattr(self, 'sgrp'):
            self.wp = []
            self.occ = []
            self.elements = []
            self.biso = []
            allwyckp = wyckpos.wp[self.sgrp]
            keys = list(allwyckp.keys())
            wpn = [int(re.sub(r"([a-z])", r"", k)) for k in keys]
            for i, (cifel, (x, y, z), occ, biso) in enumerate(self.atoms):
                el = get_element_name(cifel)
                # candidate positions from number of unique atoms
                natoms = len(self.unique_positions[i][1])
                wpcand = []
                for j, n in enumerate(wpn):
                    if n == natoms:
                        wpcand.append((keys[j], allwyckp[keys[j]]))
                for j, (k, wp) in enumerate(
                        sorted(wpcand, key=operator.itemgetter(1))):
                    parint, poslist = wp
                    for positem in poslist:
                        foundwp, xyz = testwp(parint, positem,
                                              (x, y, z), self.digits)
                        if foundwp:
                            if xyz is None:
                                self.wp.append(k)
                            else:
                                self.wp.append((k, list(xyz)))
                            self.elements.append(el)
                            self.occ.append(occ)
                            self.biso.append(biso)
                            break
                    if foundwp:
                        break
            if config.VERBOSITY >= config.INFO_ALL:
                print('XU.material: %d Wyckoff positions found for %d sites'
                      % (len(self.wp), len(self.atoms)))
            if config.VERBOSITY >= config.INFO_LOW:
                if len(self.wp) != len(self.atoms):
                    print('XU.material: missing Wyckoff positions (%d)'
                          % (len(self.atoms) - len(self.wp)))
            # free unit cell parameters
            self.crystal_system, nargs = sgl.sgrp_sym[self.sgrp_nr]
            self.crystal_system += self.sgrp_suf
            self.uc_params = []
            p2idx = {'a': 0, 'b': 1, 'c': 2,
                     'alpha': 0, 'beta': 1, 'gamma': 2}
            for pname in sgl.sgrp_params[self.crystal_system][0]:
                if pname in ('a', 'b', 'c'):
                    self.uc_params.append(self.lattice_const[p2idx[pname]])
                if pname in ('alpha', 'beta', 'gamma'):
                    self.uc_params.append(self.lattice_angles[p2idx[pname]])

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

    def SGLattice(self, use_p1=False):
        """
        create a SGLattice object with the structure from the CIF file
        """
        if not use_p1:
            if hasattr(self, 'sgrp'):
                if len(self.wp) == len(self.atoms):
                    return sgl.SGLattice(self.sgrp, *self.uc_params,
                                         atoms=self.elements, pos=self.wp,
                                         occ=self.occ, b=self.biso)
                else:
                    if config.VERBOSITY >= config.INFO_LOW:
                        print('XU.material: Wyckoff positions missing, '
                              'using P1')
            else:
                if config.VERBOSITY >= config.INFO_LOW:
                    print('XU.material: space-group detection failed, '
                          'using P1')

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

        return sgl.SGLattice(1, *itertools.chain(self.lattice_const,
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
