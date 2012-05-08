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
# Copyright (C) 2009-2010 Dominik Kriegner <dominik.kriegner@aol.at>

"""
module to handle access to the optical parameters database
"""

import tables
import numpy
import scipy.interpolate
import re


class DataBase(object):
    def __init__(self,fname):
        self.fname = fname
        self.h5file  = None  #HDF5 file object holding the database
        self.h5group = None  #Group pointing to the actual element
        self.f0_params = None
        self.f1_en = None
        self.f1    = None
        self.f2_en = None
        self.f2    = None

    def Create(self,dbname,dbdesc):
        """
        Create(dbname,dbdesc):
        Creates a new database. If the database file already exists 
        its content is delete.

        required input arguments:
        dbname .............. name of the database
        dbdesc .............. a short description of the database
        """
        #{{{
        if self.h5file!=None:
            print("database already opened - close first to create new database")
            return None

        #tryp to open the database file
        try:
            self.h5file = tables.openFile(self.fname,mode="w")
        except:
            print("cannot create database file %s!" %(self.fname))
            return None

        #set attributes to the root group with database name and 
        #description
        self.h5file.setNodeAttr("/","DBName",dbname)
        self.h5file.setNodeAttr("/","DBDesc",dbdesc)
        #}}}
        

    def Open(self,mode="r"):
        """
        Open():
        Open an existing database file.
        """
        #{{{
        if self.h5file!=None:
            print("database already opened - close first to open new database!")
            return None

        try:
            self.h5file = tables.openFile(self.fname,mode=mode)
        except:
            print("cannot open database file %s!" %(self.fname))
            return None
        #}}}

    def Close(self):
        """
        Close():
        Close an opend database file.
        """
        #{{{
        if self.h5file == None:
            print("no database file opened!")
            return None

        self.h5file.close()
        self.h5file = None
        #}}}
    
    def CreateMaterial(self,name,description):
        """
        CreateMaterial(name,description):
        This method creates a new material. If the material group already exists 
        the procedure is aborted.

        required input arguments:
        name ................... a string with the name of the material
        description ............ a string with a description of the material
        """
        #{{{
        if self.h5file == None:
            print("no database file opened!")
            return None

        try:
            g = self.getNode("/",name)
            #if this operation succeeds the material node already exists and 
            #a warning message is printed
            print("material node already exists")
            return None
        except:
            pass

        g = self.h5file.createGroup("/",name,title=description)
        #}}}

    def SetF0(self,parameters):
        """
        SetF0(parameters):
        Save f0 fit parameters for the set material. The fit parameters
        are stored in the following order:
        c,a1,b1,.......,a4,b4

        required input argument:
        parameters ............... list or numpy array with the fit parameters
        """
        #{{{
        if isinstance(parameters,list):
            p = numpy.array(parameters,dtype=numpy.float32)
        elif isinstance(parameters,numpy.ndarray):
            p = parameters.astype(numpy.float32)
        else:
            raise TypeError("f0 fit parameters must be a list or a numpy array!")

        try:
            self.h5file.removeNode(self.h5group,"f0")
        except:
            pass
    
        a = tables.Float32Atom()
        c = self.h5file.createCArray(self.h5group,"f0",a,[len(p)],title="f0 fit parameters")

        c[...] = p
        self.h5file.flush()
        #}}}

    def SetF1(self,en,f1):
        """
        SetF1(en,f1):
        Set f1 tabels values  for the active material.

        required input arguments:
        en ...................... list or numpy array with energy in (eV)
        f1 ...................... list or numpy array with f1 values
        """
        #{{{
        if isinstance(en,list):
            end = numpy.array(en,dtype=numpy.float32)
        elif isinstance(en,numpy.ndarray):
            end = en.astype(numpy.float32)
        else:
            raise TypeError("energy values must be a list or a numpy array!")

        if isinstance(f1,list):
            f1d = numpy.array(f1,dtype=numpy.float32)
        elif isinstance(f1,numpy.ndarray):
            f1d = f1.astype(numpy.float32)
        else:
            raise TypeError("f1 values must be a list or a numpy array!")

        a = tables.Float32Atom()
        
        try:
            self.h5file.removeNode(self.h5group,"en_f1")
        except:
            pass

        try:
            self.h5file.removeNode(self.h5group,"f1")
        except:
            pass

        c = self.h5file.createCArray(self.h5group,"en_f1",a,end.shape,"f1 energy scale in (eV)")
        c[...] = end
        self.h5file.flush()

        c = self.h5file.createCArray(self.h5group,"f1",a,f1d.shape,"f1 data")
        c[...] = f1d
        self.h5file.flush()
        #}}}

    def SetF2(self,en,f2):
        """
        SetF2(en,f2):
        Set f2 tabels values  for the active material.

        required input arguments:
        en ...................... list or numpy array with energy in (eV)
        f2 ...................... list or numpy array with f2 values
        """
        #{{{
        if isinstance(en,list):
            end = numpy.array(en,dtype=numpy.float32)
        elif isinstance(en,numpy.ndarray):
            end = en.astype(numpy.float32)
        else:
            raise TypeError("energy values must be a list or a numpy array!")

        if isinstance(f2,list):
            f2d = numpy.array(f2,dtype=numpy.float32)
        elif isinstance(f2,numpy.ndarray):
            f2d = f2.astype(numpy.float32)
        else:
            raise TypeError("f2 values must be a list or a numpy array!")

        a = tables.Float32Atom()
        
        try:
            self.h5file.removeNode(self.h5group,"en_f2")
        except:
            pass

        try:
            self.h5file.removeNode(self.h5group,"f2")
        except:
            pass

        c = self.h5file.createCArray(self.h5group,"en_f2",a,end.shape,"f2 energy scale in (eV)")
        c[...] = end
        self.h5file.flush()

        c = self.h5file.createCArray(self.h5group,"f2",a,f2d.shape,"f2 data")
        c[...] = f2d
        self.h5file.flush()
        #}}}

    def SetMaterial(self,name):
        """
        SetMaterial(name):
        Set a particular material in the database as the actual material. 
        All operations like setting and getting optical constants are done for
        this particular material.

        requiered input arguments:
        name ............... string with the name of the material
        """
        #{{{
        try:
            self.h5group = self.h5file.getNode("/",name)
        except:
            print("material does not exist!")

        try:
            self.f0_params = self.h5group.f0
            self.f1_en     = self.h5group.en_f1
            self.f1        = self.h5group.f1
            self.f2_en     = self.h5group.en_f2
            self.f2        = self.h5group.f2
        except:
            print("optical constants are missing!")
            #self.f0_params = None
            #self.f1_en     = None
            #self.f1        = None
            #self.f2_en     = None
            #self.f2        = None          
        #}}}

    def GetF0(self,q):
        """
        GetF0(q):
        Obtain the f0 scattering factor component for a particular 
        momentum transfer q.

        required input argument:
        q ......... single float value or numpy array
        """
        #{{{
        #get parameters
        f0_params = self.f0_params.read()
        c = f0_params[0]
        k = q/(4.*numpy.pi)
        f0 = 0.

        for i in range(1,len(f0_params)-1,2):
            a = f0_params[i]
            b = f0_params[i+1]
            f0 += a*numpy.exp(-b*k**2)

        return f0+c
        #}}}

    def GetF1(self,en):
        """
        GetF1(self,en):
        Return the second, energy dependent, real part of the scattering 
        factor for a certain energy en.

        required input arguments:
        en ............. float or numpy array with the energy
        """

        #{{{
        #check if energy is coverd by database data
        endb = self.f1_en.read()
        f1db = self.f1.read()

        if1 = scipy.interpolate.interp1d(endb,f1db,kind=1)
        f1 = if1(en)

        return f1
        #}}}

    def GetF2(self,en):
        """
        GetF2(self,en):
        Return the imaginary part of the scattering 
        factor for a certain energy en.

        required input arguments:
        en ............. float or numpy array with the energy
        """

        #{{{
        #check if energy is coverd by database data
        endb = self.f2_en.read()
        f2db = self.f2.read()

        if2 = scipy.interpolate.interp1d(endb,f2db,kind=1)
        f2 = if2(en)

        return f2
        #}}}


def init_material_db(db):
    db.CreateMaterial("H","Hydrogen")
    db.CreateMaterial("He","Helium")
    db.CreateMaterial("Li","Lithium")   
    db.CreateMaterial("Be","Berylium")
    db.CreateMaterial("B","Bor")
    db.CreateMaterial("C","Carbon")
    db.CreateMaterial("N","Nitrogen")
    db.CreateMaterial("O","Oxygen")
    db.CreateMaterial("F","Flourine")
    db.CreateMaterial("Ne","Neon")
    db.CreateMaterial("Na","Sodium")
    db.CreateMaterial("Mg","Magnesium")
    db.CreateMaterial("Al","Aluminium")
    db.CreateMaterial("Si","Silicon")
    db.CreateMaterial("P","Phosphorus")
    db.CreateMaterial("S","Sulfur")
    db.CreateMaterial("Cl","Chlorine")
    db.CreateMaterial("Ar","Argon")
    db.CreateMaterial("K","Potassium")
    db.CreateMaterial("Ca","Calcium")
    db.CreateMaterial("Sc","Scandium")
    db.CreateMaterial("Ti","Titanium")
    db.CreateMaterial("V","Vanadium")
    db.CreateMaterial("Cr","Chromium")
    db.CreateMaterial("Mn","Manganese")
    db.CreateMaterial("Fe","Iron")
    db.CreateMaterial("Co","Cobalt")
    db.CreateMaterial("Ni","Nickel")
    db.CreateMaterial("Cu","Copper")
    db.CreateMaterial("Zn","Zinc")
    db.CreateMaterial("Ga","Gallium")
    db.CreateMaterial("Ge","Germanium")
    db.CreateMaterial("As","Arsenic")
    db.CreateMaterial("Se","Selenium")
    db.CreateMaterial("Br","Bromine")
    db.CreateMaterial("Kr","Krypton")
    db.CreateMaterial("Rb","Rubidium")
    db.CreateMaterial("Sr","Strontium")
    db.CreateMaterial("Y","Yttrium")
    db.CreateMaterial("Zr","Zirconium")
    db.CreateMaterial("Nb","Niobium")
    db.CreateMaterial("Mo","Molybdenum")
    db.CreateMaterial("Tc","Technetium")
    db.CreateMaterial("Ru","Ruthenium")
    db.CreateMaterial("Rh","Rhodium")
    db.CreateMaterial("Pd","Palladium")
    db.CreateMaterial("Ag","Silver")
    db.CreateMaterial("Cd","Cadmium")
    db.CreateMaterial("In","Indium")
    db.CreateMaterial("Sn","Tin")
    db.CreateMaterial("Sb","Antimony")
    db.CreateMaterial("Te","Tellurium")
    db.CreateMaterial("I","Iodine")
    db.CreateMaterial("Xe","Xenon")
    db.CreateMaterial("Cs","Caesium")
    db.CreateMaterial("Ba","Barium")
    db.CreateMaterial("La","Lanthanum")
    db.CreateMaterial("Ce","Cerium")
    db.CreateMaterial("Pr","Praseordymium")
    db.CreateMaterial("Nd","Neodymium")
    db.CreateMaterial("Pm","Promethium")
    db.CreateMaterial("Sm","Samarium")
    db.CreateMaterial("Eu","Europium")
    db.CreateMaterial("Gd","Gadolinium")
    db.CreateMaterial("Tb","Terbium")
    db.CreateMaterial("Dy","Dysprosium")
    db.CreateMaterial("Ho","Holmium")
    db.CreateMaterial("Er","Erbium")
    db.CreateMaterial("Tm","Thulium")
    db.CreateMaterial("Yb","Ytterbium")
    db.CreateMaterial("Lu","Lutetium")
    db.CreateMaterial("Hf","Hafnium")
    db.CreateMaterial("Ta","Tantalum")
    db.CreateMaterial("W","Tungsten")
    db.CreateMaterial("Re","Rhenium")
    db.CreateMaterial("Os","Osmium")
    db.CreateMaterial("Ir","Iridium")
    db.CreateMaterial("Pt","Platinum")
    db.CreateMaterial("Au","Gold")
    db.CreateMaterial("Hg","Mercury")
    db.CreateMaterial("Tl","Thallium")
    db.CreateMaterial("Pb","Lead")
    db.CreateMaterial("Bi","Bismuth")
    db.CreateMaterial("Po","Polonium")
    db.CreateMaterial("At","Astatine")
    db.CreateMaterial("Rn","Radon")
    db.CreateMaterial("Fr","Fancium")
    db.CreateMaterial("Ra","Radium")
    db.CreateMaterial("Ac","Actinium")
    db.CreateMaterial("Th","Thorium")
    db.CreateMaterial("Pa","Protactinium")
    db.CreateMaterial("U","Urianium")



#functions to read database files
def add_f0_from_intertab(db,itabfile):
    """
    add_f0_from_intertab(db,itabfile):
    Read f0 data from international tables of crystallography and add
    it to the database.
    """

    #parse the inter. tab. file
    try:
        itf = open(itabfile,"r")
    except:
        print("cannot open f0 database file")
        return None

    #some regular expressions
    elementstr = re.compile(r"^#S")
    multiblank = re.compile(r"\s+")
    invalidelem = re.compile(r"[^A-Za-z]")
    
    while True:
        lb = itf.readline()
        if lb == "": break
        lb = lb.strip()

        if elementstr.match(lb):
            #found new element
            lb = multiblank.split(lb)
            ename = lb[2]
            #check if this is not some funny isotope
            
            if invalidelem.findall(ename)==[]:
                print("set element %s" %ename)
                db.SetMaterial(ename)
                #make one dummy read
                itf.readline()
                itf.readline()
                #read fit parameters
                lb = itf.readline()
                lb = lb.strip()
                lb = multiblank.split(lb)
                a1 = float(lb[0])
                a2 = float(lb[1])
                a3 = float(lb[2])
                a4 = float(lb[3])
                c  = float(lb[4])
                b1 = float(lb[5])
                b2 = float(lb[6])
                b3 = float(lb[7])
                b4 = float(lb[8])
                db.SetF0([c,a1,b1,a2,b2,a3,b3,a4,b4])


    itf.close()

def add_f0_from_xop(db,xopfile):
    """
    add_f0_from_xop(db,xopfile):
    Read f0 data from f0_xop.dat and add
    it to the database.
    """

    #parse the xop file
    try:
        xop = open(xopfile,"r")
    except:
        print("cannot open f0 database file")
        return None

    #some regular expressions
    elementstr = re.compile(r"^#S")
    multiblank = re.compile(r"\s+")
    invalidelem = re.compile(r"[^A-Za-z]")
    
    while True:
        lb = xop.readline()
        if lb == "": break
        lb = lb.strip()

        if elementstr.match(lb):
            #found new element
            lb = multiblank.split(lb)
            ename = lb[2]
            #check if this is not some funny isotope
            
            if invalidelem.findall(ename)==[]:
                print("set element %s" %ename)
                db.SetMaterial(ename)
                #make nine dummy reads
                xop.readline()
                xop.readline()
                xop.readline()
                xop.readline()
                xop.readline()
                xop.readline()
                xop.readline()
                xop.readline()
                xop.readline()
                #read fit parameters
                lb = xop.readline()
                lb = lb.strip()
                lb = multiblank.split(lb)
                a1 = float(lb[0])
                a2 = float(lb[1])
                a3 = float(lb[2])
                a4 = float(lb[3])
                a5 = float(lb[4])
                c  = float(lb[5])
                b1 = float(lb[6])
                b2 = float(lb[7])
                b3 = float(lb[8])
                b4 = float(lb[9])
                b5 = float(lb[10])
                db.SetF0([c,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5])

    xop.close()


def add_f1f2_from_henkedb(db,henkefile):
    """
    Read f1 and f2 data from Henke database and add
    it to the database.
    """

    #parse the inter. tab. file
    try:
        hf = open(henkefile,"r")
    except:
        print("cannot open f1f2 database file")
        return None

    #some regular expressions
    elementstr = re.compile(r"^#S")
    multiblank = re.compile(r"\s+")
    invalidelem = re.compile(r"[^A-Za-z]")
    
    while True:
        lb = hf.readline()
        if lb == "": break
        lb = lb.strip()

        if elementstr.match(lb):
            #found new element
            lb = multiblank.split(lb)
            enum = lb[1]
            ename = lb[2]
            #check if this is not some funny isotope
            
            if invalidelem.findall(ename)==[]:
                print("set element %s"%ename)
                db.SetMaterial(ename)
                #make one dummy read
                for i in range(5): hf.readline()
                
                #read data
                en_list = []
                f1_list = []
                f2_list = []
                while True:
                    lb = hf.readline()
                    lb = lb.strip()
                    lb = multiblank.split(lb)
                    en = float(lb[0])
                    f1 = float(lb[1])-float(enum) #to account for wrong f1 definition in Henke db
                    f2 = float(lb[2])
                    en_list.append(en)
                    f1_list.append(f1)
                    f2_list.append(f2)
                    if en==30000.:
                        db.SetF1(en_list,f1_list)
                        db.SetF2(en_list,f2_list)
                        break

    hf.close()


def add_f1f2_from_kissel(db,kisselfile):
    """
    Read f1 and f2 data from Henke database and add
    it to the database.
    """

    #parse the f1f2 file
    try:
        kf = open(kisselfile,"r")
    except:
        print("cannot open f1f2 database file")
        return None

    #some regular expressions
    elementstr = re.compile(r"^#S")
    multiblank = re.compile(r"\s+")
    invalidelem = re.compile(r"[^A-Za-z]")
    
    while True:
        lb = kf.readline()
        if lb == "": break
        lb = lb.strip()

        if elementstr.match(lb):
            #found new element
            lb = multiblank.split(lb)
            enum = lb[1]
            ename = lb[2]
            #check if this is not some funny isotope
            
            if invalidelem.findall(ename)==[]:
                print("set element %s"%ename)
                db.SetMaterial(ename)
                #make 28 dummy reads
                for i in range(28): kf.readline()
                
                #read data
                en_list = []
                f1_list = []
                f2_list = []
                while True:
                    lb = kf.readline()
                    lb = lb.strip()
                    lb = multiblank.split(lb)
                    try:
                        en = float(lb[0])*1000 # convert energy
                        f1 = float(lb[4])-float(enum) #to account for wrong f1 definition in Henke db
                        f2 = float(lb[5])
                        en_list.append(en)
                        f1_list.append(f1)
                        f2_list.append(f2)
                        if en==10000000.:
                            db.SetF1(en_list,f1_list)
                            db.SetF2(en_list,f2_list)
                            break
                    except: 
                        print(lb)
                        break

    kf.close()
