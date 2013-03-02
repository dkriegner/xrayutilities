============
sphinx4scons
============

:author: Orlando Wingbrant

Introduction
============

The sphinx4scons package provides an SCons_ tool to invoke the Sphinx_
documentation generator from SCons build scripts.

It has been tested with SCons v2.0.1 and Sphinx v1.1.2

.. _SCons: http://www.scons.org
.. _Sphinx: http://sphinx.pocoo.org

Provided builder
================

``Sphinx()``

``env.Sphinx()``

  The Sphinx builder is similar to the Java builder in that the source
  and the target arguments are directories instead of single files.

  The builder will scan the given source directory for source
  files. It will read the sphinx configuration file for the project
  and deduce the extension used for source files from there. The
  builder also honors the ``exclude_patterns`` entry in the
  configuration file.

  The return value from the Sphinx builder is dependent upon the
  documentation format being generated. For example, if the output is
  a single html file, the name of that file will be the return
  value. If LaTeX or man files are being generated, the names of these
  files will be in the return value. File names for different formats
  are fetched from applicable sections in the sphinx configuration
  file.

  The builder requires at least two arguments, the first is the target
  directory, and the second is the source directory. There is also a
  set of optional builder arguments:

  ``builder``
    A string with the name of the documentation generator (sphinx
    builder) to use. Please see sphinx documentation for available
    builders. The default sphinx builder to use is the html builder.

  ``config``
    Path to the sphinx configuration file to use. 

  ``doctree``
    Path to doctrees directory.

  ``options``
    A string with ``sphinx_build`` options. It will be copied verbatim
    to the command line.

  ``settings``
    A python dictionary with strings. Each entry will be put on the
    command line as ``-D "key=value"`` and will hence override settings
    in the sphinx configuration file.

  ``tags``
    A python list of strings. Each string will be preceded by ``-t``
    on the command line, and hence define a tag.

  Examples:
  
    ``Sphinx('_build/html', '.')``

    This will generate documentation using the sphinx html
    documentation generator and the current directory as source and
    put the resulting files in the ``_build/html`` directory.

    ``Sphinx('_build/latex', '.', builder='latex')``
    
    This will generate latex documentation (provided that it is set up
    properly in the sphinx configuration file) in the ``_build/latex``
    directory.

    ``Sphinx('_build/doc/dirhtml', 'doc', builder='dirhtml',``
              ``tags=['draft'], config='doc/draft')``

    This will generate a dirhtml directory structure (see sphinx
    documentation) in the ``_build/doc/dirhtml`` directory, using
    ``doc`` as the source directory and the file ``doc/draft/conf.py``
    as the sphinx configuration file. Additionaly it will set the tag
    "draft" on the sphinx command line.


Construction environment variables
==================================

The sphinx4scons tool sets and uses the following set of construction
set variables.

SPHINXBUILD
  The sphinx build script. Default is ``sphinx-build``

SPHINXBUILDER
  The sphinx builder to use if no other option is given, defaults to
  ``html``.

SPHINXCOM
  The command line used to invoke sphinx. Default is ``$SPHINXBUILD
  $_SPHINXOPTIONS ${SOURCE.attributes.root}
  ${TARGET.attributes.root}``

SPHINXCOMSTR
  This only affects presentation. If set to a non-empty value this
  string will be displayed when the sphinx command is invoked, instead
  of the content of the $SPHINXCOM variable.

SPHINXCONFIG
  Path to sphinx configuration file. The default is "" (the empty
  string) which will make sphinx use the file ``conf.py`` in the
  source directory.

SPHINXDOCTREE
  Directory for doctrees. The default is "" (the empty string) which
  will make sphinx fallback to its default value, ``.doctrees`` in the
  target directory.

SPHINXFLAGS
  Additional command-line flags, will be copied verbatim to the
  command line. The default is "" (the empty string).

_SPHINXOPTIONS
  Generated from SPHINXFLAGS, SPHINXTAGS, SPHINXSETTINGS,
  SPHINXDOCTREE, SPINXBUILDER, and various arguments to the builder.

SPHINXSETTINGS
  This construction variable is a python dictionary with strings. Each
  key-value entry will be put on the command line preceded by -D,
  e.g. the dictionary ``{"key":"value"}``, will be transformed to ``-D
  "key=value"``. It is used to override settings in the configuration
  file. The default is {} (the empty dictionary).

SPHINXTAGS
  This construction variable is a python list of strings. Each entry
  will be put on the command line as a tag definition, i.e. preceded
  by -t. The default is [] (the empty list).


Known issues
============

This module supports building documentation from what scons calls
repositories (don't confuse that with source code control
repositories). However, due to the way sphinx-build works, all
document source files must be in the same directory tree, not part of
it in a local diectory and part of it in a repository tree. You can
however have the configuration file and files and directories relative
to that in a repository and source files in e.g. a local directory. 
