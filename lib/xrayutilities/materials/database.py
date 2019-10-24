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
# Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>
# Copyright (C) 2009-2016 Dominik Kriegner <dominik.kriegner@gmail.com>

"""
module to handle the access to the optical parameters database
"""

import re

import h5py
import numpy
import scipy.constants


class DataBase(object):

    def __init__(self, fname):
        self.fname = fname
        self.h5file = None  # HDF5 file object holding the database
        self.h5group = None  # Group pointing to the actual element
        self.f0_params = None
        self.f1_en = None
        self.f1 = None
        self.f2_en = None
        self.f2 = None
        self.weight = None
        self.color = None
        self.radius = numpy.nan
        self.matname = None

    def Create(self, dbname, dbdesc):
        """
        Creates a new database. If the database file already exists
        its content is delete.

        Parameters
        ----------
        dbname :    str
            name of the database
        dbdesc :    str
            a short description of the database
        """
        if self.h5file is not None:
            print("database already opened - "
                  "close first to create new database")
            return None

        # tryp to open the database file
        try:
            self.h5file = h5py.File(self.fname, 'w')
        except OSError:
            print('cannot create database file %s!' % (self.fname))
            raise

        # set attributes to the root group with database name and
        # description
        self.h5file.attrs['DBName'] = dbname
        self.h5file.attrs['DBDesc'] = dbdesc

    def Open(self, mode='r'):
        """
        Open an existing database file.
        """
        if self.h5file is not None:
            print('database already opened - '
                  'close first to open new database!')
            return

        try:
            self.h5file = h5py.File(self.fname, mode)
        except OSError:
            print("cannot open database file %s!" % (self.fname))

    def Close(self):
        """
        Close an opend database file.
        """
        if self.h5file is None:
            print("no database file opened!")
            return

        self.h5file.close()
        self.h5file = None

    def CreateMaterial(self, name, description):
        """
        This method creates a new material. If the material group already
        exists the procedure is aborted.

        Parameters
        ----------
        name :          str
            name of the material
        description :   str
            description of the material
        """
        if self.h5file is None:
            print("no database file opened!")
            return

        if name in self.h5file:
            # if the material node already exists a warning message is printed
            print("material node already exists")
        else:
            g = self.h5file.create_group(name)
            g.attrs['name'] = description

    def SetWeight(self, weight):
        """
        Save weight of the element as float

        Parameters
        ----------
        weight :    float
            atomic standard weight of the element
        """
        if not isinstance(weight, float):
            raise TypeError("weight parameter must be a float!")

        self.h5group.attrs['atomic_standard_weight'] = weight
        self.h5file.flush()

    def SetColor(self, color):
        """
        Save color of the element for visualization

        Parameters
        ----------
        color :    tuple, str
            matplotlib color for the element
        """
        if not isinstance(color, (tuple, str)):
            raise TypeError("color parameter must be a tuple or str!")

        self.h5group.attrs['color'] = color
        self.h5file.flush()

    def SetRadius(self, radius):
        """
        Save atomic radius for visualization

        Parameters
        ----------
        radius:     float
            atomic radius in Angstrom
        """
        if not isinstance(radius, float):
            raise TypeError("radius parameter must be a float!")

        self.h5group.attrs['atomic_radius'] = radius
        self.h5file.flush()

    def SetF0(self, parameters, subset='default'):
        """
        Save f0 fit parameters for the set material. The fit parameters
        are stored in the following order:
        c, a1, b1,......., a4, b4

        Parameters
        ----------
        parameters :    list or array-like
            fit parameters
        subset :        str, optional
            name the f0 dataset
        """
        if isinstance(parameters, list):
            p = numpy.array(parameters, dtype=numpy.float32)
        elif isinstance(parameters, numpy.ndarray):
            p = parameters.astype(numpy.float32)
        else:
            raise TypeError("f0 fit parameters must be a "
                            "list or a numpy array!")

        if not subset:
            subset = 'default'

        try:
            del self.h5group['f0/%s' % subset]
        except KeyError:
            pass

        self.h5group.create_dataset('f0/%s' % subset, data=p)
        self.h5file.flush()

    def SetF1F2(self, en, f1, f2):
        """
        Set f1, f2 values for the active material.

        Parameters
        ----------
        en :    list or array-like
            energy in (eV)
        f1 :    list or array-like
            f1 values
        f2 :    list or array-like
            f2 values
        """
        if isinstance(en, (list, tuple)):
            end = numpy.array(en, dtype=numpy.float32)
        elif isinstance(en, numpy.ndarray):
            end = en.astype(numpy.float32)
        else:
            raise TypeError("energy values must be a list or a numpy array!")

        if isinstance(f1, (list, tuple)):
            f1d = numpy.array(f1, dtype=numpy.float32)
        elif isinstance(f1, numpy.ndarray):
            f1d = f1.astype(numpy.float32)
        else:
            raise TypeError("f1 values must be a list or a numpy array!")

        if isinstance(f2, (list, tuple)):
            f2d = numpy.array(f2, dtype=numpy.float32)
        elif isinstance(f2, numpy.ndarray):
            f2d = f2.astype(numpy.float32)
        else:
            raise TypeError("f2 values must be a list or a numpy array!")

        try:
            del self.h5group['en_f12']
        except KeyError:
            pass

        try:
            del self.h5group['f1']
        except KeyError:
            pass

        try:
            del self.h5group['f2']
        except KeyError:
            pass

        self.h5group.create_dataset('en_f12', data=end)
        self.h5group.create_dataset('f1', data=f1d)
        self.h5group.create_dataset('f2', data=f2d)
        self.h5file.flush()

    def SetMaterial(self, name):
        """
        Set a particular material in the database as the actual material.  All
        operations like setting and getting optical constants are done for this
        particular material.

        Parameters
        ----------
        name :  str
            name of the material
        """
        if self.matname == name:
            return
        try:
            self.h5group = self.h5file[name]
        except KeyError:
            print("XU.materials.database: material '%s' not existing!" % name)

        try:
            self.f0_params = self.h5group['f0']
        except KeyError:
            self.f0_params = None
        try:
            self.f1_en = self.h5group['en_f12']
            self.f1 = self.h5group['f1']
        except KeyError:
            self.f1_en = None
            self.f1 = None
        try:
            self.f2_en = self.h5group['en_f12']
            self.f2 = self.h5group['f2']
        except KeyError:
            self.f2_en = None
            self.f2 = None
        try:
            self.weight = self.h5group.attrs['atomic_standard_weight']
        except KeyError:
            self.weight = None
        try:
            self.radius = self.h5group.attrs['atomic_radius']
        except KeyError:
            self.radius = numpy.nan
        try:
            self.color = self.h5group.attrs['color']
        except KeyError:
            self.color = None
        self.matname = name

    def GetF0(self, q, dset='default'):
        """
        Obtain the f0 scattering factor component for a particular
        momentum transfer q.

        Parameters
        ----------
        q :     float or array-like
            momentum transfer
        dset :  str, optional
            specifies which dataset (different oxidation states)
            should be used
        """
        # get parameters from file
        if not dset:
            dset = 'default'
        f0_params = self.f0_params[dset]
        # calculate f0
        if isinstance(q, (numpy.ndarray, list, tuple)):
            ql = numpy.asarray(q)
            f0 = f0_params[0] * numpy.ones(ql.shape)
        else:
            ql = q
            f0 = f0_params[0]
        k = ql / (4. * numpy.pi)

        for i in range(1, len(f0_params) - 1, 2):
            a = f0_params[i]
            b = f0_params[i + 1]
            f0 += a * numpy.exp(-b * k ** 2)

        return f0

    def GetF1(self, en):
        """
        Return the second, energy dependent, real part of the scattering
        factor for a certain energy en.

        Parameters
        ----------
        en :    float or array-like
            energy
        """
        if1 = numpy.interp(en, self.f1_en, self.f1,
                           left=numpy.nan, right=numpy.nan)

        return if1

    def GetF2(self, en):
        """
        Return the imaginary part of the scattering
        factor for a certain energy en.

        Parameters
        ----------
        en :    float or array-like
            energy
        """
        if2 = numpy.interp(en, self.f2_en, self.f2,
                           left=numpy.nan, right=numpy.nan)

        return if2


def init_material_db(db):
    db.CreateMaterial("dummy", "Dummy atom")
    db.CreateMaterial("H", "Hydrogen")
    db.CreateMaterial("D", "Deuterium")
    db.CreateMaterial("T", "Tritium")
    db.CreateMaterial("He", "Helium")
    db.CreateMaterial("Li", "Lithium")
    db.CreateMaterial("Be", "Berylium")
    db.CreateMaterial("B", "Bor")
    db.CreateMaterial("C", "Carbon")
    db.CreateMaterial("N", "Nitrogen")
    db.CreateMaterial("O", "Oxygen")
    db.CreateMaterial("F", "Flourine")
    db.CreateMaterial("Ne", "Neon")
    db.CreateMaterial("Na", "Sodium")
    db.CreateMaterial("Mg", "Magnesium")
    db.CreateMaterial("Al", "Aluminium")
    db.CreateMaterial("Si", "Silicon")
    db.CreateMaterial("P", "Phosphorus")
    db.CreateMaterial("S", "Sulfur")
    db.CreateMaterial("Cl", "Chlorine")
    db.CreateMaterial("Ar", "Argon")
    db.CreateMaterial("K", "Potassium")
    db.CreateMaterial("Ca", "Calcium")
    db.CreateMaterial("Sc", "Scandium")
    db.CreateMaterial("Ti", "Titanium")
    db.CreateMaterial("V", "Vanadium")
    db.CreateMaterial("Cr", "Chromium")
    db.CreateMaterial("Mn", "Manganese")
    db.CreateMaterial("Fe", "Iron")
    db.CreateMaterial("Co", "Cobalt")
    db.CreateMaterial("Ni", "Nickel")
    db.CreateMaterial("Cu", "Copper")
    db.CreateMaterial("Zn", "Zinc")
    db.CreateMaterial("Ga", "Gallium")
    db.CreateMaterial("Ge", "Germanium")
    db.CreateMaterial("As", "Arsenic")
    db.CreateMaterial("Se", "Selenium")
    db.CreateMaterial("Br", "Bromine")
    db.CreateMaterial("Kr", "Krypton")
    db.CreateMaterial("Rb", "Rubidium")
    db.CreateMaterial("Sr", "Strontium")
    db.CreateMaterial("Y", "Yttrium")
    db.CreateMaterial("Zr", "Zirconium")
    db.CreateMaterial("Nb", "Niobium")
    db.CreateMaterial("Mo", "Molybdenum")
    db.CreateMaterial("Tc", "Technetium")
    db.CreateMaterial("Ru", "Ruthenium")
    db.CreateMaterial("Rh", "Rhodium")
    db.CreateMaterial("Pd", "Palladium")
    db.CreateMaterial("Ag", "Silver")
    db.CreateMaterial("Cd", "Cadmium")
    db.CreateMaterial("In", "Indium")
    db.CreateMaterial("Sn", "Tin")
    db.CreateMaterial("Sb", "Antimony")
    db.CreateMaterial("Te", "Tellurium")
    db.CreateMaterial("I", "Iodine")
    db.CreateMaterial("Xe", "Xenon")
    db.CreateMaterial("Cs", "Caesium")
    db.CreateMaterial("Ba", "Barium")
    db.CreateMaterial("La", "Lanthanum")
    db.CreateMaterial("Ce", "Cerium")
    db.CreateMaterial("Pr", "Praseordymium")
    db.CreateMaterial("Nd", "Neodymium")
    db.CreateMaterial("Pm", "Promethium")
    db.CreateMaterial("Sm", "Samarium")
    db.CreateMaterial("Eu", "Europium")
    db.CreateMaterial("Gd", "Gadolinium")
    db.CreateMaterial("Tb", "Terbium")
    db.CreateMaterial("Dy", "Dysprosium")
    db.CreateMaterial("Ho", "Holmium")
    db.CreateMaterial("Er", "Erbium")
    db.CreateMaterial("Tm", "Thulium")
    db.CreateMaterial("Yb", "Ytterbium")
    db.CreateMaterial("Lu", "Lutetium")
    db.CreateMaterial("Hf", "Hafnium")
    db.CreateMaterial("Ta", "Tantalum")
    db.CreateMaterial("W", "Tungsten")
    db.CreateMaterial("Re", "Rhenium")
    db.CreateMaterial("Os", "Osmium")
    db.CreateMaterial("Ir", "Iridium")
    db.CreateMaterial("Pt", "Platinum")
    db.CreateMaterial("Au", "Gold")
    db.CreateMaterial("Hg", "Mercury")
    db.CreateMaterial("Tl", "Thallium")
    db.CreateMaterial("Pb", "Lead")
    db.CreateMaterial("Bi", "Bismuth")
    db.CreateMaterial("Po", "Polonium")
    db.CreateMaterial("At", "Astatine")
    db.CreateMaterial("Rn", "Radon")
    db.CreateMaterial("Fr", "Fancium")
    db.CreateMaterial("Ra", "Radium")
    db.CreateMaterial("Ac", "Actinium")
    db.CreateMaterial("Th", "Thorium")
    db.CreateMaterial("Pa", "Protactinium")
    db.CreateMaterial("U", "Urianium")
    db.CreateMaterial("Np", "Neptunium")
    db.CreateMaterial("Pu", "Plutonium")
    db.CreateMaterial("Am", "Americium")
    db.CreateMaterial("Cm", "Curium")
    db.CreateMaterial("Bk", "Berkelium")
    db.CreateMaterial("Cf", "Californium")
    db.CreateMaterial("Es", "Einsteinium")
    db.CreateMaterial("Fm", "Fermium")
    db.CreateMaterial("Md", "Mendelevium")
    db.CreateMaterial("No", "Nobelium")
    db.CreateMaterial("Lr", "Lawrencium")
    db.CreateMaterial("Rf", "Rutherfordium")
    db.CreateMaterial("Db", "Dubnium")
    db.CreateMaterial("Sg", "Seaborgium")
    db.CreateMaterial("Bh", "Bohrium")
    db.CreateMaterial("Hs", "Hassium")
    db.CreateMaterial("Mt", "Meitnerium")
    db.CreateMaterial("Ds", "Darmstadtium")
    db.CreateMaterial("Rg", "Roentgenium")
    db.CreateMaterial("Cn", "Copernicium")
    db.CreateMaterial("Nh", "Nihonium")
    db.CreateMaterial("Fl", "Flerovium")
    db.CreateMaterial("Mc", "Moscovium")
    db.CreateMaterial("Lv", "Livermorium")
    db.CreateMaterial("Ts", "Tennessine")
    db.CreateMaterial("Og", "Oganesson")


# functions to read database files
def add_f0_from_intertab(db, itf, verbose=False):
    """
    Read f0 data from International Tables of Crystallography and add
    it to the database.
    """
    # some regular expressions
    elementstr = re.compile(r"^#S")
    multiblank = re.compile(r"\s+")
    while True:
        lb = itf.readline().decode("utf-8")
        if lb == "":
            break
        lb = lb.strip()

        if elementstr.match(lb):
            # found new element
            lb = multiblank.split(lb)

            # determine oxidation state and element name
            elemstate = re.sub('[A-Za-z]', '', lb[2])
            for r, o in zip(('dot', 'p', 'm'), ('.', '+', '-')):
                elemstate = elemstate.replace(o, r)
            if elemstate == 'p2':  # fix wrong name in the source file
                elemstate = '2p'
            ename = re.sub('[^A-Za-z]', '', lb[2])

            if verbose:
                print("{pyname} = Atom('{name}', {num})".format(
                    pyname=ename+elemstate, name=lb[2], num=lb[1]))
            db.SetMaterial(ename)
            # make two dummy reads
            for i in range(2):
                itf.readline()
            # read fit parameters
            lb = itf.readline().decode("utf-8")
            lb = lb.strip()
            lb = multiblank.split(lb)
            a1 = float(lb[0])
            a2 = float(lb[1])
            a3 = float(lb[2])
            a4 = float(lb[3])
            c = float(lb[4])
            b1 = float(lb[5])
            b2 = float(lb[6])
            b3 = float(lb[7])
            b4 = float(lb[8])
            db.SetF0([c, a1, b1, a2, b2, a3, b3, a4, b4], subset=elemstate)


def add_f0_from_xop(db, xop, verbose=False):
    """
    Read f0 data from f0_xop.dat and add
    it to the database.
    """
    # some regular expressions
    elementstr = re.compile(r"^#S")
    multiblank = re.compile(r"\s+")

    while True:
        lb = xop.readline().decode("utf-8")
        if lb == "":
            break
        lb = lb.strip()

        if elementstr.match(lb):
            # found new element
            lb = multiblank.split(lb)
            # determine oxidation state and element name
            elemstate = re.sub('[A-Za-z]', '', lb[2])
            for r, o in zip(('dot', 'p', 'm'), ('.', '+', '-')):
                elemstate = elemstate.replace(o, r)
            ename = re.sub('[^A-Za-z]', '', lb[2])

            if verbose:
                print("{pyname} = Atom('{name}', {num})".format(
                    pyname=ename+elemstate, name=lb[2], num=lb[1]))
            db.SetMaterial(ename)

            # make nine dummy reads
            for i in range(9):
                xop.readline()
            # read fit parameters
            lb = xop.readline().decode("utf-8")
            lb = lb.strip()
            lb = multiblank.split(lb)
            a1 = float(lb[0])
            a2 = float(lb[1])
            a3 = float(lb[2])
            a4 = float(lb[3])
            a5 = float(lb[4])
            c = float(lb[5])
            b1 = float(lb[6])
            b2 = float(lb[7])
            b3 = float(lb[8])
            b4 = float(lb[9])
            b5 = float(lb[10])
            db.SetF0([c, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5])


def add_f1f2_from_henkedb(db, hf, verbose=False):
    """
    Read f1 and f2 data from Henke database and add
    it to the database.
    """
    # some regular expressions
    elementstr = re.compile(r"^#S")
    multiblank = re.compile(r"\s+")
    invalidelem = re.compile(r"[^A-Za-z]")

    while True:
        lb = hf.readline().decode("utf-8")
        if lb == "":
            break
        lb = lb.strip()

        if elementstr.match(lb):
            # found new element
            lb = multiblank.split(lb)
            enum = lb[1]
            ename = lb[2]
            # check if this is not some funny isotope

            if invalidelem.findall(ename) == []:
                if verbose:
                    print("set element %s" % ename)
                db.SetMaterial(ename)
                # make one dummy read
                for i in range(5):
                    hf.readline()

                # read data
                en_list = []
                f1_list = []
                f2_list = []
                while True:
                    lb = hf.readline().decode("utf-8")
                    lb = lb.strip()
                    lb = multiblank.split(lb)
                    en = float(lb[0])
                    # to account for wrong f1 definition in Henke db
                    f1 = float(lb[1]) - float(enum)
                    f2 = float(lb[2])
                    en_list.append(en)
                    f1_list.append(f1)
                    f2_list.append(f2)
                    if en == 30000.:
                        db.SetF1F2(en_list, f1_list, f2_list)
                        break


def add_f1f2_from_kissel(db, kf, verbose=False):
    """
    Read f1 and f2 data from Henke database and add
    it to the database.
    """
    # some regular expressions
    elementstr = re.compile(r"^#S")
    multiblank = re.compile(r"\s+")
    invalidelem = re.compile(r"[^A-Za-z]")

    while True:
        lb = kf.readline().decode("utf-8")
        if lb == "":
            break
        lb = lb.strip()

        if elementstr.match(lb):
            # found new element
            lb = multiblank.split(lb)
            enum = lb[1]
            ename = lb[2]
            # check if this is not some funny isotope

            if invalidelem.findall(ename) == []:
                if verbose:
                    print("set element %s" % ename)
                db.SetMaterial(ename)
                # make 28 dummy reads
                for i in range(28):
                    kf.readline()

                # read data
                en_list = []
                f1_list = []
                f2_list = []
                while True:
                    lb = kf.readline().decode("utf-8")
                    lb = lb.strip()
                    lb = multiblank.split(lb)
                    en = float(lb[0]) * 1000  # convert energy
                    # to account for wrong f1 definition in Henke db
                    f1 = float(lb[4]) - float(enum)
                    f2 = float(lb[5])
                    en_list.append(en)
                    f1_list.append(f1)
                    f2_list.append(f2)
                    if en == 10000000.:
                        db.SetF1F2(en_list, f1_list, f2_list)
                        break


def add_f1f2_from_ascii_file(db, asciifile, element, verbose=False):
    """
    Read f1 and f2 data for specific element from ASCII file (3 columns) and
    save it to the database.
    """

    # parse the f1f2 file
    try:
        af = numpy.loadtxt(asciifile)
    except OSError:
        print("cannot open f1f2 database file")
        return None
    db.SetMaterial(element)

    en = af[:, 0]
    f1 = af[:, 1]
    f2 = af[:, 2]
    db.SetF1F2(en, f1, f2)


def add_mass_from_NIST(db, nistfile, verbose=False):
    """
    Read atoms standard mass and save it to the database.
    The mass of the natural isotope mixture is taken from the NIST data!
    """
    # some regular expressions
    isotope = re.compile(r"^Atomic Number =")
    standardw = re.compile(r"^Standard Atomic Weight")
    relativew = re.compile(r"^Relative Atomic Mass")
    number = re.compile(r"[0-9.]+")
    multiblank = re.compile(r"\s+")

    # parse the nist file
    with open(nistfile, "r") as nf:
        while True:
            lb = nf.readline()
            if lb == "":
                break
            lb = lb.strip()

            if isotope.match(lb):
                # found new element
                lb = multiblank.split(lb)
                lb = nf.readline()
                lb = lb.strip()
                lb = multiblank.split(lb)
                ename = lb[-1]

                if verbose:
                    print("set element %s" % ename)
                db.SetMaterial(ename)

                # read data
                while True:
                    lb = nf.readline()
                    lb = lb.strip()
                    if relativew.match(lb):
                        lb = multiblank.split(lb)
                        # extract fallback weight
                        w = float(number.findall(lb[-1])[0])
                        db.SetWeight(w * scipy.constants.atomic_mass)
                    elif standardw.match(lb):
                        lb = multiblank.split(lb)
                        # extract average weight
                        try:
                            w = float(number.findall(lb[-1])[0])
                            db.SetWeight(w * scipy.constants.atomic_mass)
                        except IndexError:
                            pass
                        break


def add_color_from_JMOL(db, cfile, verbose=False):
    """
    Read color from JMOL color table and save it to the database.
    """
    with open(cfile, "r") as f:
        for line in f.readlines():
            s = line.split()
            ename = s[1]
            color = [float(num)/255. for num in s[2].strip('[]').split(',')]
            color = tuple(color)
            if verbose:
                print("set element %s" % ename)
            db.SetMaterial(ename)
            db.SetColor(color)


def add_radius_from_WIKI(db, dfile, verbose=False):
    """
    Read radius from Wikipedia radius table and save it to the database.
    """
    with open(dfile, "r") as f:
        for line in f.readlines():
            s = line.split(',')
            ename = s[1]
            radius = float(s[3]) / 100.
            if verbose:
                print("set element %s" % ename)
            db.SetMaterial(ename)
            db.SetRadius(radius)
