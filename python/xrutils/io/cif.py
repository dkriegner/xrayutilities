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
# Copyright (C) 2010 Dominik Kriegner <dominik.kriegner@aol.at>

import re
import numpy

from .. import materials
from .. import config

re_loop = re.compile(r"^loop_")
re_symop = re.compile(r"^_space_group_symop_operation_xyz")
re_atom = re.compile(r"^_atom_site_label")
re_labelline = re.compile(r"^_")
re_emptyline = re.compile(r"^\s*$")
re_cell_a = re.compile(r"^_cell_length_a")
re_cell_b = re.compile(r"^_cell_length_b")
re_cell_c = re.compile(r"^_cell_length_c")
re_cell_alpha = re.compile(r"^_cell_angle_alpha")
re_cell_beta = re.compile(r"^_cell_angle_beta")
re_cell_gamma = re.compile(r"^_cell_angle_gamma")

class CIFFile(object):
    """
    class for parsing CIF (Crystallographic Information File) files. The class aims
    to provide an additional way of creating material classes instead of manual entering
    of the information the lattice constants and unit cell structure are parsed from the
    CIF file
    """
    def __init__(self,filename):
        #{{{
        """
        initialization of the CIFFile class

        Parameter
        ---------
         filename:  filename of the CIF file
        """
        self.filename = filename
        self.digits = 3 # number of digits used to check if position is unique

        try:
            self.fid = open(self.filename,"r")
        except:
            raise IOError("cannot open CIF file %s" %self.filename)

        self.Parse()

        self.SymStruct()
        #}}}

    def __del__(self):
        #{{{
        """
        class destructor which closes open files
        """
        if not self.fid == None:
            self.fid.close()
        #}}}

    def Parse(self):
        #{{{
        """
        function to parse a CIF file. The function reads the
        space group symmetry operations and the basic atom positions
        as well as the lattice constants and unit cell angles
        """

        self.symops = []
        self.atoms = []
        self.lattice_const = numpy.zeros(3,dtype=numpy.double)
        self.lattice_angles = numpy.zeros(3,dtype=numpy.double)

        self.fid.seek(0) # set file pointer to the beginning
        loop_start = False
        symop_loop = False
        atom_loop = False

        for line in self.fid.readlines():
            if config.VERBOSITY >= config.DEBUG:
                print(line)
            if re_loop.match(line): # start of loop
                loop_start = True
                symop_loop = False
                atom_loop = False
            elif re_symop.match(line) and loop_start: # start of symmetry op. loop
                symop_loop = True
                loop_start = False
            elif re_atom.match(line) and loop_start: # start of atom position loop
                atom_loop = True
                loop_start = False
            elif re_labelline.match(line): # label line, check if needed
                if re_cell_a.match(line):
                    self.lattice_const[0] = float(line.split()[1])
                elif re_cell_b.match(line):
                    self.lattice_const[1] = float(line.split()[1])
                elif re_cell_c.match(line):
                    self.lattice_const[2] = float(line.split()[1])
                elif re_cell_alpha.match(line):
                    self.lattice_angles[0] = float(line.split()[1])
                elif re_cell_beta.match(line):
                    self.lattice_angles[1] = float(line.split()[1])
                elif re_cell_gamma.match(line):
                    self.lattice_angles[2] = float(line.split()[1])

            elif re_emptyline.match(line):
                continue
            elif symop_loop: # symmetry operation entry
                opstr = line.split()[0]
                opstr = re.sub(r"^'",r"(",opstr)
                opstr = re.sub(r"'$",r")",opstr)
                opstr = re.sub(r"/([1-9])",r"/\1.",opstr) # add a comma to a fraction
                self.symops.append(opstr)
            elif atom_loop: # atom label and position
                asplit = line.split()
                alabel = asplit[0]
                apos = (float(asplit[1]),float(asplit[2]),float(asplit[3]))
                self.atoms.append((alabel,apos))
        #}}}

    def SymStruct(self):
        #{{{
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
            el = re.sub(r"([1-9])",r"",a[0])
            for symop in self.symops:
                exec("pos = numpy.array("+ symop+ ")")
                # check that position is within unit cell
                pos = pos - pos//1
                # check if position is unique
                unique = True
                for upos in unique_pos:
                    if (numpy.round(upos,self.digits) == numpy.round(pos,self.digits)).all():
                        unique = False
                if unique:
                    unique_pos.append(pos)
            exec("element = materials.elements."+el)
            self.unique_positions.append((element, unique_pos))
        #}}}

    def Lattice(self):
        #{{{
        """
        returns a lattice object with the structure from the CIF file
        """

        lb = materials.LatticeBase()
        for atom in self.unique_positions:
            element = atom[0]
            for pos in atom[1]:
                lb.append(element,pos)

        #unit cell vectors
        ca = numpy.cos(numpy.radians(self.lattice_angles[0]))
        cb = numpy.cos(numpy.radians(self.lattice_angles[1]))
        cg = numpy.cos(numpy.radians(self.lattice_angles[2]))
        sa = numpy.sin(numpy.radians(self.lattice_angles[0]))
        sb = numpy.sin(numpy.radians(self.lattice_angles[1]))
        sg = numpy.sin(numpy.radians(self.lattice_angles[2]))

        a1 = self.lattice_const[0]*numpy.array([1,0,0],dtype=numpy.double)
        a2 = self.lattice_const[1]*numpy.array([cg,sg,0],dtype=numpy.double)
        a3 = self.lattice_const[2]*numpy.array([cb , (ca-cb*cg)/sg , numpy.sqrt(1-ca**2-  cb**2-cg**2+2*ca*cb*cg)/sg],dtype=numpy.double)
        # create lattice
        l = materials.Lattice(a1,a2,a3,base=lb)

        return l
        #}}}

    def __str__(self):
        #{{{
        """
        returns a string with positions and names of the atoms
        """
        ostr = ""
        ostr += "unit cell structure\n"
        ostr += "a: %8.4f b: %8.4f c: %8.4f\n" %tuple(self.lattice_const)
        ostr += "alpha: %6.2f beta: %6.2f gamma: %6.2f\n" %tuple(self.lattice_angles)
        ostr += "Unique atom positions in unit cell\n"
        for atom in self.unique_positions:
            ostr += atom[0].name + " (%d): \n" %atom[0].num
            for pos in atom[1]:
                ostr += str(numpy.round(pos,self.digits)) + "\n"
        return ostr
        #}}}
