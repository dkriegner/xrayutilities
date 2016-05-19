.. highlight:: python
   :linenothreshold: 5

.. _simulationspage:

Simulation examples
===================

In the following a few code-snippets are shown which should help you getting started with reflectivity and diffraction simulations using *xrayutilities*. All simulations in *xrayutilities* are for layers systems and currently there are no plans to extend this to other geometries. Note that not all of the codes shown in the following will be run-able as stand-alone scripts. For fully running scripts look in the ``examples`` directory in the download found `here <https://sourceforge.net/projects/xrayutilities>`_.

Building Layer stacks for simulations
-------------------------------------

The basis of all simulations in *xrayutilities* are stacks of layers. Therefore several functions exist to build up such layered systems. The basic building block of all of them is a :class:`~xrayutilities.simpack.smaterials.Layer` object which takes a material and its thickness in ångström as initializing parameter.::

    import xrayutilities as xu
    lay = xu.simpack.Layer(xu.materials.Si, 200)

In the shown example a silicon layer with 20 nm thickness is created. The first argument is the material of the layer. For diffraction simulations this needs to be derived from the :class:`~xrayutilities.materials.material.Crystal`-class. This means all predefined materials in *xrayutitities* can be used for this purpose. For x-ray reflectivity simulations, however, also knowing the chemical composition and density of the material is sufficient.

A 5 nm thick metallic CoFe compound layer can therefore be defined by::


    rho_cf = 0.5*8900 + 0.5*7874  # mass density in kg/m^3
    mCoFe = xu.materials.Amorphous('CoFe', rho_cf, [('Co', 0.5), ('Fe', 0.5)])
    lCoFe = xu.simpack.Layer(mat_cf, 50)

.. note:: The :class:`~xrayutilities.simpack.smaterials.Layer` object can have several more model dependent properties discussed in detail below.

When several layers are defined they can be combined to a :class:`~xrayutilities.simpack.smaterials.LayerStack` which is used for the simulations below.::

    sub = xu.simpack.Layer(xu.materials.Si, inf)
    lay1 = xu.simpack.Layer(xu.materials.Ge, 200)
    lay2 = xu.simpack.Layer(xu.materials.SiO2, 30)
    ls = xu.simpack.LayerStack('Si/Ge', sub, lay1, lay2)
    # or equivalently
    ls = xu.simpack.LayerStack('Si/Ge', sub + lay1 + lay2)

The last two lines show two different options of creating a stack of layers. As is shown in the last example the substrate thickness can be infinite (see below) and layers can be also stacked by summation. For creation of more complicated superlattice stacks one can further use multiplication::

    lay1 = xu.simpack.Layer(xu.materials.SiGe(0.3), 50)
    lay2 = xu.simpack.Layer(xu.materials.SiGe(0.6), 40)
    ls = xu.simpack.LayerStack('Si/SiGe SL', sub + 5*(lay1 + lay2))

Pseudomorphic Layers
~~~~~~~~~~~~~~~~~~~~

All stacks of layers described above use the materials in the layer as they are supplied. However, epitaxial systems often adopt the inplane lattice parameter of the layers beneath. To mimic this behavior you can either supply the :class:`~xrayutilities.simpack.smaterials.Layer` objects which custom :class:`~xrayutilities.materials.material.Crystal` objects which have the appropriate lattice parameters or use the :class:`~xrayutilities.simpack.PseudomorphicStack*` classes which to the adaption of the lattice parameters automatically. In this respect the 'relaxation' parameter of the :class:`~xrayutilities.simpack.smaterials.Layer` class is important since it allows to create partially/fully relaxed layers.::

    sub = xu.simpack.Layer(xu.materials.Si, inf)
    buf1 = xu.simpack.Layer(xu.materials.SiGe(0.5), 5000, relaxation=1.0)
    buf2 = xu.simpack.Layer(xu.materials.SiGe(0.8), 5000, relaxation=1.0)
    lay1 = xu.simpack.Layer(xu.materials.SiGe(0.6), 50, relaxation=0.0)
    lay2 = xu.simpack.Layer(xu.materials.SiGe(1.0), 50, relaxation=0.0)
    # create pseudomorphic superlattice stack
    pls = xu.simpack.PseudomorphicStack001('SL 5/5', sub+buf1+buf2+5*(lay1+lay2))

.. note:: As indicated by the function name the PseudomorphicStack currently only works for (001) surfaces and cubic materials. Implementations for other surface orientations are planned.

If you would like to check the resulting lattice objects of the different layers you could use::

    for l in pls:
        print(l.material.lattice)

Special layer types
~~~~~~~~~~~~~~~~~~~

So far one special layer mimicking a layer with gradually changing chemical composition is implemented. It consists of several thin sublayers of constant composition. So in order to obtain a smooth grading one has to select enough sublayers. This however has a negativ impact on the performance of all simulation models. A tradeoff needs to found! Below a graded SiGe buffer is shown which consists of 100 sublayers and has total thickness of 1µm.::

    buf = xu.simpack.GradedLayerStack(xu.materials.SiGe,
                                      0.2,  # xfrom Si0.8Ge0.2
                                      0.7,  # xto Si0.3Ge0.7
                                      100,  # number of sublayers
                                      10000,  # total thickness
                                      relaxation=1.0)


Setting up a model
------------------

This sectiondescribes the parameters which are common for all diffraction models in *xrayutilties*-``simpack``. All models need a list of Layers for which the reflected/diffracted signal will be calculated. Further all models have some common parameters which allow scaling and background addition in the model output and contain general information about the calculation which are model-independent. These are

 * 'experiment': an :class:`~xrayutilities.experiment.Experiment`/:class:`~xrayutilities.experiment.HXRD` object which defines the surface geometry of the model. If none is given a default class with (001) surface is generated.
 * 'resolution_width': width of the Gaussian resolution function used to convolute with the data. The unit of this parameters depends on the model and can be either in degree or 1/\AA.
 * 'I0': is the primary beam flux/intensity
 * 'background': is the background added to the simulation after it was scaled by I0
 * 'energy': energy in eV used to obtain the optical parameters for the simulation. The energy can alternatively also be supplied via the 'experiment' parameter, however, the 'energy' value overrules this setting. If no energy is given the default energy from the configuration is used.

The mentioned parameters can be supplied to the constructor method of all model classes derived from :class:`~xrayutilities.simpack.models.LayerModel`, which applies to all examples mentioned below.::

    m = xu.simpack.SpecularReflectivityModel(layerstack, I0=1e6, background=1,
                                             resolution_width=0.001)

Reflectivity calculation and fitting
------------------------------------

Currently only the Parrat formalism including non-correlated roughnesses is included for specular x-ray reflectivity calculations. A minimal working example for a reflectivity calculation follows.::

    # building a stack of layers
    sub = xu.simpack.Layer(xu.materials.GaAs, inf, roughness=2.0)
    lay1 = xu.simpack.Layer(xu.materials.AlGaAs(0.25), 75, roughness=2.5)
    lay2 = xu.simpack.Layer(xu.materials.AlGaAs(0.75), 25, roughness=3.0)
    pls = xu.simpack.PseudomorphicStack001('pseudo', sub+5*(lay1+lay2))

    # reflectivity calculation
    m = xu.simpack.SpecularReflectivityModel(pls, sample_width=5, beam_width=0.3)
    ai = linspace(0, 5, 10000)
    Ixrr = m.simulate(ai)

In addition to the layer thickness also the roughness and relative density of a Layer can be set since they are important for the reflectivity calculation. This can be done upon definition of the :class:`~xrayutilities.simpack.smaterials.Layer` or also manipulated at any later stage.
Such x-ray reflectivity calculations can also be fitted to experimental data using the :func:`~xrayutilities.simpack.fit.fit_xrr` function which is shown in detail in the example below (which is also included in the example directory). The fitting is performed using the `lmfit <https://lmfit.github.io/lmfit-py/>`_ Python package which needs to be installed when you want to use this fitting function. This package allows to build complicated models including bounds and correlations between parameters.

.. code-block:: python

    from matplotlib.pylab import *
    import xrayutilities as xu
    import lmfit
    import numpy

    # load experimental data
    ai, edata, eps = numpy.loadtxt('data/xrr_data.txt'), unpack=True)
    ai /= 2.0

    # define layers
    # SiO2 / Ru(5) / CoFe(3) / IrMn(3) / AlOx(10)
    lSiO2 = xu.simpack.Layer(xu.materials.SiO2, inf)
    lRu = xu.simpack.Layer(xu.materials.Ru, 50)
    rho_cf = 0.5*8900 + 0.5*7874
    mat_cf = xu.materials.Amorphous('CoFe', rho_cf, [('Co', 0.5), ('Fe', 0.5)])
    lCoFe = xu.simpack.Layer(mat_cf, 30)
    lIrMn = xu.simpack.Layer(xu.materials.Ir20Mn80, 30)
    lAl2O3 = xu.simpack.Layer(xu.materials.Al2O3, 100)

    m = xu.simpack.SpecularReflectivityModel(lSiO2, lRu, lCoFe, lIrMn, lAl2O3,
                                             energy='CuKa1')

    p = lmfit.Parameters()
    #          (Name                ,     Value,  Vary,   Min,  Max, Expr)
    p.add_many(('SiO2_thickness'    , numpy.inf, False,  None, None, None),
               ('SiO2_roughness'    ,       2.5,  True,     0,    8, None),
               ('Ru_thickness'      ,      47.0,  True,    25,   70, None),
               ('Ru_roughness'      ,       2.8,  True,     0,    8, None),
               ('Ru_density'        ,       1.0,  True,   0.8,  1.0, None),
               ('CoFe_thickness'    ,      27.0,  True,    15,   50, None),
               ('CoFe_roughness'    ,       4.6,  True,     0,    8, None),
               ('CoFe_density'      ,       1.0,  True,   0.8,  1.2, None),
               ('Ir20Mn80_thickness',      21.0,  True,    15,   40, None),
               ('Ir20Mn80_roughness',       3.0,  True,     0,    8, None),
               ('Ir20Mn80_density'  ,       1.1,  True,   0.8,  1.2, None),
               ('Al2O3_thickness'   ,     100.0,  True,    70,  130, None),
               ('Al2O3_roughness'   ,       5.5,  True,     0,    8, None),
               ('Al2O3_density'     ,       1.0,  True,   0.8,  1.2, None),
               ('I0'                ,    6.75e9,  True,   3e9,  8e9, None),
               ('background'        ,        81,  True,    40,  100, None),
               ('sample_width'      ,       6.0, False,     2,    8, None),
               ('beam_width'        ,      0.25, False,   0.2,  0.4, None),
               ('resolution_width'  ,      0.02, False,  0.01, 0.05, None))

    res = xu.simpack.fit_xrr(m, p, ai, data=edata, eps=eps, xmin=0.05, xmax=8.0,
                             plot=True, verbose=True)
    lmfit.report_fit(res, min_correl=0.5)

This script can interactively show the fitting progress and after the fitting shows the final plot including the x-ray reflectivity trace of the initial and final parameters.

.. figure:: pics/xrr_fitting.svg
   :alt: XRR fitting output
   :width: 400 px

   The picture shows the final plot of the fitting example shown in one of the example scripts.

After building a :class:`~xrayutilities.simpack.models.SpecularReflectivityModel` is built or fitted the density profile resulting from the thickness and roughness of layers can be plotted easily by::

    m.densityprofile(500, plot=True)  # 500 number of points

.. figure:: pics/xrr_densityprofile.svg
   :alt: XRR density profile resulting from the XRR fit shown above
   :width: 300 px

Diffraction calculation
-----------------------

From the very same models as used for XRR calculation one can also perform crystal truncation rod simulations around certain Bragg peaks using various different diffraction models. Depending on the system to model you will have to choose the most apropriate model. Below a short description of the implemented models is given followed by two examples.

Kinematical diffraction models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most basic models consider only the kinematic diffraction of layers and substrate. Especially the semiinfinite substrate is not well described using the kinematical approximation which results in considerable deviations in close vicinity to substrate Bragg peak with respect to the more acurate dynamical diffraction models.

Such a basic model is employed by::

    mk = xu.simpack.KinematicalModel(pls, energy=en, resolution_width=0.0001)
    Ikin = mk.simulate(qz, hkl=(0, 0, 4))

A more appealing kinematical model is represented by the :class:`~xrayutilities.simpack.models.KinematicalMultiBeamModel` class which implements a true multibeam theory is, however, restricted to the use of (001) surfaces and layer thicknesses will be changed to be a multiple of the out of plane lattice spacing. This is necessary since otherwise the structure factor of the unit cell can not be used for the calculation.

It can be employed by::

    mk = xu.simpack.KinematicalMultiBeamModel(pls, energy=en,
                                              surface_hkl=(0, 0, 1),
                                              resolution_width=0.0001)
    Imult = mk.simulate(qz, hkl=(0, 0, 4))

This model is expected to provide good results especially far away from the substrate peak where the influence of other Bragg peaks on the truncation rod and the variation of the structure factor can not be neglected.

Both kinematical model's :func:`~xrayutilities.simpack.models.KinematicalMultiBeamModel.simulate` method offers two keyword arguments with which basic absorption and refraction correction can be added to the basic models.

.. note:: The kinematical models can also handle a semi-infinitely thick substrate which results in a diverging intensity at the Bragg peak but provides a basic description of the substrates truncation rod.

Dynamical diffraction models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Acurate description of the diffraction from thin films in close vicinity to the diffraction signal from a bulk substrate is only possible using the dynamical diffraction theory. In **xrayutilities** the dynamical two-beam theory with 4 tiepoints for the calculation of the dispersion surface is implemented. To use this theory you have to supply the :func:`~xrayutilities.simpack.models.DynamicalModel.simulate` method with the incidence angle in degree. Accordingly the 'resolution_width' parameter is also in degree for this model.::

    md = xu.simpack.DynamicalModel(pls, energy=en, resolution_width=resol)
    Idyn = md.simulate(ai, hkl=(0, 0, 4))

A second simplified dynamical model (:class:`~xrayutilities.simpack.models.SimpleDynamicalCoplanarModel`) is also implemented should, however, not be used since its approximations cause mistakes in almost all relevant cases.

The :class:`~xrayutilities.simpack.models.DynamicalModel` supports the calculation of diffracted signal for 'S' and 'P' polarization geometry. To simulate diffraction data of laboratory sources with Ge(220) monochromator crystal one should use::

    qGe220 = linalg.norm(xu.materials.Ge.Q(2, 2, 0))
    thMono = arcsin(qGe220 * lam / (4*pi))
    md = xu.simpack.DynamicalModel(pls, energy='CuKa1',
                                   Cmono=cos(2 * thMono),
                                   polarization='both')
    Idyn = md.simulate(ai, hkl=(0, 0, 4))


Comparison of diffraction models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below we show the different implemented models for the case of epitaxial GaAs/AlGaAs and Si/SiGe bilayers. These two cases have very different separation of the layer Bragg peak from the substrate and therefore provide good model system for our models.

We will compare the (004) Bragg peak calculated with different models and but otherwise equal parameters. For scripts used to perform the shown calculation you are referred to the ``examples`` directory.

.. figure:: pics/xrd_algaas004.svg
   :alt: (004) of AlGaAs(100nm) on GaAs
   :width: 400 px

   XRD simulations of the (004) Bragg peak of ~100 nm AlGaAs on GaAs(001) using various diffraction models

.. figure:: pics/xrd_sige004.svg
   :alt: (004) of SiGe(15nm) on Si
   :width: 400 px

   XRD simulations of the (004) Bragg peak of 15 nm Si\ :sub:`0.4` Ge\ :sub:`0.6` on Si(001) using various diffraction models

As can be seen in the images we find that for the AlGaAs system all models except the very basic kinematical model yield an very similar diffraction signal. The second kinematic diffraction model considering the contribution of multiple Bragg peaks on the same truncation rod fails to describe only the ratio of substrate and layer signal, but otherwise results in a very similar line shape as the traces obtained by the dynamic theory.

For the SiGe/Si bilayer system bigger differences between the kinematic and dynamic models are found. Further also the difference between the simpler and more sophisticated dynamic model gets obvious further away from the reference position. Interestingly also the multibeam kinematic theory differs considerable from the best dynamic model. As is evident from this second comparison the correct choice of model for the particular system under condideration is crucial for comparison with experimental data.