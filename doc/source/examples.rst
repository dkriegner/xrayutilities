.. highlight:: python
   :linenothreshold: 5

.. _examplespage:

Examples
========

In the following a few code-snippets are shown which should help you getting started with *xrayutilities*. Not all of the codes shown in the following will be run-able as stand-alone script. For fully running scripts look in the ``examples`` directory in the download found `here <https://sourceforge.net/projects/xrayutilities>`_.


Reading data from data files
----------------------------

The ``io`` submodule provides classes for reading x-ray diffraction data in
various formats. In the following few examples are given.

Reading SPEC files
^^^^^^^^^^^^^^^^^^

Working with spec files in *xrayutilities* can be done in two distinct ways. 
 1. parsing the spec file for scan headers; and parsing the data only when needed
 2. parsing the spec file for scan headers; parsing all data and dump them to an HDF5 file; reading the data from the HDF5 file. 

Both methods have their pros and cons. For example when you parse the spec-files over a network connection you need to re-read the data again over the network if using method 1) whereas you can dump them to a local file with method 2). But you will parse data of the complete file while dumping it to the HDF5 file. 

Both methods work incremental, so they do not start at the beginning of the file when you reread it, but start from the last position they were reading and work with files including data from linear detectors.

An working example for both methods is given in the following.::

    import tables
    import xrayutilities as xu
    import os
    
    # open spec file or use open SPECfile instance
    try: s
    except NameError:
        s = xu.io.SPECFile("sample_name.spec",path="./specdir")
    
    # method (1)
    s.scan10.ReadData()
    scan10data = s.scan10.data
    
    # method (2)
    h5file = os.path.join("h5dir","h5file.h5")
    s.Save2HDF5(h5file) # save content of SPEC file to HDF5 file
    # read data from HDF5 file
    [angle1,angle2],scan10data = xu.io.geth5_scan(h5file,[10], "motorname1", "motorname2")


.. seealso::
   the fully working example :ref:`helloworld`

In the following it is shown how to re-parsing the SPEC file for new scans and reread the scans (1) or update the HDF5 file(2)

::

    s.Update() # reparse for new scans in open SPECFile instance
    
    # reread data method (1)
    s.scan10.ReadData()
    scan10data = s.scan10.data 
    
    # reread data method (2)
    s.Save2HDF5(h5) # save content of SPEC file to HDF5 file
    # read data from HDF5 file
    [angle1,angle2],scan10data = xu.io.geth5_scan(h5file,[10], "motorname1", "motorname2")


Reading EDF files
^^^^^^^^^^^^^^^^^

EDF files are mostly used to store CCD frames at ESRF recorded from various different detectors. This format is therefore used in combination with SPEC files. In an example the EDFFile class is used to parse the data from EDF files and store them to an HDF5 file. HDF5 if perfectly suited because it can handle large amount of data and compression.::

    import tables 
    import xrayutilities as xu
    import numpy
    
    specfile = "specfile.spec"
    h5file = "h5file.h5"
    h5 = tables.openFile(h5file,mode='a')
    
    s = xu.io.SPECFile(specfile,path=specdir)
    s.Save2HDF5(h5) # save to hdf5 file
    
    # read ccd frames from EDF files
    for i in range(1,1000,1):
        efile = "edfdir/sample_%04d.edf" %i
        e = xu.io.edf.EDFFile(efile,path=specdir)
        e.ReadData()
        g5 = h5.createGroup(h5.root,"frelon_%04d" %i)
        e.Save2HDF5(h5,group=g5)
    
    h5.close()

.. seealso::
   the fully working example provided in the ``examples`` directory perfectly suited for reading data from beamline ID01


Other formats
^^^^^^^^^^^^^

Other formats which can be read include

 * files recorded from `Panalytical <http://www.panalytical.com>`_ diffractometers in the ``.xrdml`` format. 
 * files produces by the experimental control software at Hasylab/Desy (spectra).
 * ccd images in the tiff file format produced by RoperScientific CCD cameras and Perkin Elmer detectors.
 * files from recorded by Seifert diffractometer control software (``.nja``)
 * basic support is also provided for reading of ``cif`` files from structure database to extract unit cell parameters

See the ``examples`` directory for more information and working example scripts.

Angle calculation using ``experiment`` and ``material`` classes
---------------------------------------------------------------

Methods for high angle x-ray diffraction experiments. Mostly for experiments performed in coplanar scattering geometry. An example will be given for the calculation of the position of Bragg reflections.

::

    import xrayutilities as xu
    Si = xu.materials.Si  # load material from materials submodule
    
    # initialize experimental class with directions from experiment
    hxrd = xu.HXRD(Si.Q(1,1,-2),Si.Q(1,1,1))
    # calculate angles of Bragg reflections and print them to the screen
    om,chi,phi,tt = hxrd.Q2Ang(Si.Q(1,1,1))
    print("Si (111)")
    print("om,tt: %8.3f %8.3f" %(om,tt))
    om,chi,phi,tt = hxrd.Q2Ang(Si.Q(2,2,4))
    print("Si (224)")
    print("om,tt: %8.3f %8.3f" %(om,tt))

Note that on line 5 the ``HXRD`` class is initialized without specifying the energy used in the experiment. It will use the default energy stored in the configuration file, which defaults to CuK-alpha1.

One could also call::

    hxrd = xu.HXRD(Si.Q(1,1,-2),Si.Q(1,1,1),en=10000) # energy in eV

to specify the energy explicitly.
The ``HXRD`` class by default describes a four-circle goniometer as described in more detail `here <http://www.certif.com/spec_manual/fourc_4_1.html>`_.

Similar functions exist for other experimental geometries. For grazing incidence diffraction one might use::

    gid = xu.GID(Si.Q(1,-1,0),Si.Q(0,0,1))
    # calculate angles and print them to the screen
    (alphai,azimuth,tt,beta) = gid.Q2Ang(Si.Q(2,-2,0))
    print("azimuth,tt: %8.3f %8.3f" %(azimuth,tt))

There is on implementation of a GID 2S+2D diffractometer. Be sure to check if the order of the detector circles fits your goniometer, otherwise define one yourself!

There exists also a powder diffraction class, which is able to convert powder scans from angular to reciprocal space and furthermore powder scans of materials can be simulated in a very primitive way, which should only be used to get an idea of the peak positions expected from a certain material.

::

    import xrayutilities as xu
    import matplotlib.pyplot as plt
    
    energy = (2*8048 + 8028)/3. # copper k alpha 1,2
    
    # creating Indium powder 
    In_powder = xu.Powder(xu.materials.In,en=energy)
    # calculating the reflection strength for the powder
    In_powder.PowderIntensity()
    
    # convoluting the peaks with a gaussian in q-space
    peak_width = 0.01 # in q-space
    resolution = 0.0005 # resolution in q-space
    In_th,In_int = In_powder.Convolute(resolution,peak_width)
    
    plt.figure()
    plt.xlabel(r"2Theta (deg)"); plt.ylabel(r"Intensity")
    # plot the convoluted signal
    plt.plot(In_th*2,In_int/In_int.max(),'k-',label="Indium powder convolution")
    # plot each peak in a bar plot
    plt.bar(In_powder.ang*2, In_powder.data/In_powder.data.max(), width=0.3, bottom=0, 
            linewidth=0, color='r',align='center', orientation='vertical',label="Indium bar plot")
    
    plt.legend(); plt.set_xlim(15,100); plt.grid()

One can also print the peak positions and other informations of a powder by

 >>> print In_powder
    Powder diffraction object 
    -------------------------
    Material: In
    Lattice:
    a1 = (3.252300 0.000000 0.000000), 3.252300
    a2 = (0.000000 3.252300 0.000000), 3.252300
    a3 = (0.000000 0.000000 4.946100), 4.946100
    alpha = 90.000000, beta = 90.000000, gamma = 90.000000
    Lattice base:
    Base point 0: In (49) (0.000000 0.000000 0.000000) occ=1.00 b=0.00
    Base point 1: In (49) (0.500000 0.500000 0.500000) occ=1.00 b=0.00
    Reflections: 
    --------------
          h k l     |    tth    |    |Q|    |    Int     |   Int (%)
       ---------------------------------------------------------------
        [-1, 0, -1]    32.9611      2.312       217.75      100.00
         [0, 0, -2]    36.3267      2.541        41.80       19.20
        [-1, -1, 0]    39.1721      2.732        67.72       31.10
       [-1, -1, -2]    54.4859      3.731        50.75       23.31
       ....

Using the ``Gridder`` classes
-----------------------------

*xrayutilities* provides Gridder classes for 1D, 2D, and 3D data sets. These Gridders map irregular spaced data onto a regular grid. 
This is often needed after transforming data measured at equally spaced angular positions to reciprocal space were their spacing is irregular.

In 1D this process actually equals the calculation of a histogramm. 
Below you find the most basic way of using the Gridder in 2D. Other dimensions work very similar.

The most easiest use (what most user might need) is:

::
    import xrayutilities as xu # import python package
    g = xu.Gridder2D(100,101) # initialize the Gridder object, which will 
    # perform Gridding to a regular grid with 100x101 points
    #====== load some data here =====
    g(x,y,data) # call the gridder with the data
    griddata = g.data # the data object of the Gridder contains the gridded data.

_.. note: previously you could use the Gridder's gdata object, which was always an internal buffer and should not be used anymore!

A more complicated example showing also sequential gridding is shown below. You need sequential gridding when you can not load all data at the same time, which is often problematic with 3D data sets. In such cases you need to specify the data range before the first call to the gridder. 

::
    import xrayutilities as xu # import python package
    g = xu.Gridder2D(100,101) # initialize the Gridder object
    g.KeepData(True)
    g.dataRange((1,2),(3,4))  # (xgrd_min,xgrd_max),(ygrd_min,ygrd_max)
    #====== load some data here =====
    g(x,y,data) # call the gridder with the data
    griddata = g.data # the data object of the Gridder contains the so far gridded data.

    #====== load some more data here =====
    g(x,y,data) # call the gridder with the new data
    griddata = g.data # the data object of the Gridder contains the combined gridded data.


Using the ``material`` class
----------------------------

*xrayutilities* provides a set of python classes to describe crystal lattices and 
materials.

Examples show how to define a new material by defining its lattice and deriving a new material, furthermore materials can be used to calculate the structure factor of a Bragg reflection for an specific energy or the energy dependency of its structure factor for anomalous scattering. Data for this are taken from a database which is included in the download.

First defining a new material from scratch is shown. This consists of an lattice with base and the type of atoms with elastic constants of the material::

    import xrayutilities as xu
    
    # defining a ZincBlendeLattice with two types of atoms and lattice constant a
    def ZincBlendeLattice(aa,ab,a):
        #create lattice base
        lb = xu.materials.LatticeBase()
        lb.append(aa,[0,0,0])
        lb.append(aa,[0.5,0.5,0])
        lb.append(aa,[0.5,0,0.5])
        lb.append(aa,[0,0.5,0.5])
        lb.append(ab,[0.25,0.25,0.25])
        lb.append(ab,[0.75,0.75,0.25])
        lb.append(ab,[0.75,0.25,0.75])
        lb.append(ab,[0.25,0.75,0.75])
                
        #create lattice vectors
        a1 = [a,0,0]
        a2 = [0,a,0]
        a3 = [0,0,a]
                
        l = xu.materials.Lattice(a1,a2,a3,base=lb)    
        return l
    
    # defining InP, no elastic properties are given, 
    # helper functions exist to create the (6,6) elastic tensor for cubic materials 
    atom_In = xu.materials.elements.In
    atom_P = xu.materials.elements.P
    elastictensor = xu.materials.CubicElasticTensor(10.11e+10,5.61e+10,4.56e+10)
    InP  = xu.materials.Material("InP",ZincBlendeLattice(atom_In, atom_P ,5.8687), elastictensor)

InP is of course already included in the xu.materials module and can be loaded by::

    InP = xu.materials.InP

like many other materials.


Using the material properties the calculation of the reflection strength of a Bragg reflection can be done as follows::

    import xrayutilities as xu
    import numpy
    
    # defining material and experimental setup
    InAs = xu.materials.InAs
    energy= 8048 # eV
    
    # calculate the structure factor for InAs (111) (222) (333)
    hkllist = [[1,1,1],[2,2,2],[3,3,3]]
    for hkl in hkllist:
        qvec = InAs.Q(hkl)
        F = InAs.StructureFactor(qvec,energy)
        print(" |F| = %8.3f" %numpy.abs(F))


Similar also the energy dependence of the structure factor can be determined::

    import matplotlib.pyplot as plt
    
    energy= numpy.linspace(500,20000,5000) # 500 - 20000 eV
    F = InAs.StructureFactorForEnergy(InAs.Q(1,1,1),energy)
    
    plt.figure(); plt.clf()
    plt.plot(energy,F.real,'k-',label='Re(F)')
    plt.plot(energy,F.imag,'r-',label='Imag(F)')
    plt.xlabel("Energy (eV)"); plt.ylabel("F"); plt.legend()



It is also possible to calculate the components of the structure factor of atoms, which may be needed for input into XRD simulations.::

    # f = f0(|Q|) + f1(en) + j * f2(en)
    import xrayutilities as xu
    import numpy
    
    Fe = xu.materials.elements.Fe # iron atom
    Q = numpy.array([0,0,1.9],dtype=numpy.double)
    en = 10000 # energy in eV
    
    print "Iron (Fe): E: %9.1f eV" % en
    print "f0: %8.4g" % Fe.f0(numpy.linalg.norm(Q))
    print "f1: %8.4g" % Fe.f1(en)
    print "f2: %8.4g" % Fe.f2(en)


Calculation of diffraction angles for a general geometry
--------------------------------------------------------

Often the restricted predefined geometries are not corresponding to the experimental setup, nevertheless *xrayutilities* is able to calculate the goniometer angles needed to reach a certain reciprocal space position.

For this purpose the goniometer together with the geometric restrictions need to be defined and the q-vector in laboratory reference frame needs to be specified.
This works for arbitrary goniometer, however, the user is expected to set up bounds to put restrictions to the number of free angles to obtain reproducible results. 
In general only three angles are needed to fit an arbitrary q-vector (2 sample + 1 detector angles or
1 sample + 2 detector). 

The example below shows the necessary code to perform such an angle calculation for a costum defined material with orthorhombic unit cell.

.. code-block:: python

    import xrayutilities as xu
    import numpy as np

    def Pnma(a,b,c):
        #create orthorhombic unit cell
        l = xu.materials.Lattice([a,0,0],[0,b,0],[0,0,c])
        return l

    latticeConstants=[5.600,7.706,5.3995]
    SmFeO3 = xu.materials.Material("SmFeO3",Pnma(*latticeConstants))

    qconv=xu.QConversion(('x+','z+'),('z+','x+'),(0,1,0)) # 2S+2D goniometer 
    hxrd = xu.HXRD(SmFeO3.Q(0,0,1),SmFeO3.Q(1,1,0),qconv=qconv) # [1,1,0] surface normal

    hkl=(2,0,0)
    q_material = SmFeO3.Q(hkl)
    q_laboratory = hxrd.Transform(q_material) # transform 

    print('SmFeO3: \thkl ', hkl, '\tqvec ', np.round(q_material,5))
    print('Lattice plane distance: %.4f'%SmFeO3.planeDistance(hkl))

    #### determine the goniometer angles with the correct geometry restrictions
    # tell bounds of angles / (min,max) pair or fixed value for all motors
    # maximum of three free motors! here incidence angle fixed to 5 degree
    # om, phi, tt, delta
    bounds = (5,(-180,180),(-1,90),(-1,90))
    ang,qerror,errcode = xu.Q2AngFit(q_laboratory,hxrd,bounds)
    print('err %d (%.3g) angles %s'%(errcode,qerror,str(np.round(ang,5))))  # check that qerror is small!!
    print('sanity check with back-transformation (hkl): ',np.round(hxrd.Ang2HKL(*ang,mat=SmFeO3),5))


User-specific config file
-------------------------

Several options of *xrayutilities* can be changed by options in a config file. This includes the default x-ray energy as well as parameters to set the number of threads used by the parallel code and the verbosity of the output.

The default options are stored inside the installad Python module and should not be changed. Instead it is suggested to use a user-specific config file
'~/.xrayutilities.conf' or a 'xrayutilities.conf' file in the working directory. 

An example of such a user config file is shown below:

.. code-block:: python

    # begin of xrayutilities configuration
    [xrayutilities]

    # verbosity level of information and debugging outputs
    #   0: no output
    #   1: very import notes for users
    #   2: less import notes for users (e.g. intermediate results)
    #   3: debuging output (e.g. print everything, which could be interesing)
    #   levels can be changed in the config file as well
    verbosity = 1

    # default wavelength in Angstrom, 
    wavelength = MoKa1 # Molybdenum K alpha1 radiation (17479.374eV)

    # default energy in eV
    # if energy is given wavelength settings will be ignored
    #energy = 10000 #eV

    # number of threads to use in parallel sections of the code
    nthreads = 1
    #   0: the maximum number of available threads will be used (as returned by omp_get_max_threads())
    #   n: n-threads will be used 



Determining detector parameters
-------------------------------

In the following three examples of how to determine the detector parameters for linear and area detectors is given.
The procedure we use is in more detail described in this `article <http://arxiv.org/abs/1304.1732>`_.

Linear detectors
^^^^^^^^^^^^^^^^

To determine the detector parameters of a linear detector one needs to perform a scan with the detector angle through the primary beam and aquire a detector spectrum at any point.

Using the following script determines the parameters necessary for the detector initialization, which are:

* pixelwidth of one channel
* the center channel
* and the detector tilt (optional)

.. literalinclude:: example_xu_linear_detector_parameters.py
    :linenos:
    :language: python


Area detector (Variant 1)
^^^^^^^^^^^^^^^^^^^^^^^^^

To determine the detector parameters of a area detector one needs to perform scans with the detector angles through the primary beam and aquire a detector images at any position.
For the area detector at least two scans (one with the outer detector and and one with the inner detector angle) are required.

Using the following script determines the parameters necessary for the detector initialization from such scans in the primary beam only. Further down we discuss an other variant which is also able to use additionally detector images recorded at the Bragg reflection of a known reference crystal.

The determined detector parameters are:

* pixelwidth of the channels in both directions (2 parameters)
* center channels: position of the primary beam at the true zero position of the goniometer (considering the outer angle offset) (2 parameters)
* detector tilt azimuth in degree from 0 to 360
* detector tilt angle in degree (>0deg)
* detector rotation around the primary beam in degree
* outer angle offset, which describes a offset of the outer detector angle from its true zero position

The misalignment parameters can be fixed during the fitting.

.. literalinclude:: example_xu_ccd_parameter.py
    :linenos:
    :language: python

A possible output of this script could be

.. code-block:: python

    fitted parameters: epsilon: 8.0712e-08 (2,['Parameter convergence'])
    param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset) 
    param: 140.07 998.34 4.4545e-05 4.4996e-05 72.0 1.97 -0.792 -1.543
    please check the resulting data (consider setting plot=True)
    detector rotation axis / primary beam direction (given by user): ['z+', 'y-'] / x+
    detector pixel directions / distance: z- y+ / 1
    detector initialization with: init_area('z-','y+',cch1=140.07,cch2=998.34,Nch1=516,Nch2=516, pwidth1=4.4545e-05,pwidth2=4.4996e-05,distance=1.,detrot=-0.792,tiltazimuth=72.0,tilt=1.543)
    AND ALWAYS USE an (additional) OFFSET of -1.9741deg in the OUTER DETECTOR ANGLE!


The output gives the fitted detector parameters and compiles the python code line one needs to use to initialize the detector.
Important to note is that the outer angle offset which was determined by the fit (-1.9741 degree in the aboves example) is not included in the initialization of the detector parameters BUT needs to be used in every call to the q-conversion function as offset. 
This step needs to be performed manually by the user!

Area detector (Variant 2)
^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to scans in the primary beam this variant enables also the use of detector images recorded in scans at Bragg reflections of a known reference materials. However this also required that the sample orientation and x-ray wavelength need to be fit.
To keep the additional parameters as small as possible we only implemented this for symmetric coplanar diffractions. 

The advantage of this method is that it is more sensitive to the outer angle offset also at large detector distances.
The additional parameters are:

* sample tilt angle in degree
* sample tilt azimuth in degree
* and the x-ray wavelength in Angstrom

.. literalinclude:: example_xu_ccd_parameter_hkl.py
    :linenos:
    :language: python

