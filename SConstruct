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
# Copyright (C) 2010-2012 Dominik Kriegner <dominik.kriegner@aol.at>

import os
import datetime
import subprocess

AddOption("--prefix",dest="prefix",type="string",
          default=os.sys.prefix,metavar="INSTALL_ROOT",
          action="store",nargs=1,help='installation prefix under which all files will be installed.')

env = Environment(PREFIX=GetOption("prefix"),
                  ENV=os.environ,
                  DESTDIR=os.path.expandvars('${DESTDIR}'),
                  CCFLAGS=["-fPIC","-Wall","-std=c99"],
                  tools=["default", "disttar"], toolpath=[os.path.join(".","tools")],
                  LIBS=["m"])

vars = Variables()
vars.Add(PathVariable("DESTDIR",'Destination variable (prepended to prefix)',None,PathVariable.PathAccept))
vars.Update(env)

if "win" in os.sys.platform:
    Tool('mingw')(env)

# create correct destdir install prefix
# is only needed/used on linux/darwin systems
if env['DESTDIR'] == "" or env['DESTDIR'] == "${DESTDIR}":
    env['DESTDIRPREFIX'] = env['PREFIX']
else:
    env['DESTDIRPREFIX'] = os.path.join(env['DESTDIR'],env['PREFIX'][1:])

#add the aliases for install target
if os.sys.platform in ["darwin","linux2"]:
    env.Alias("install",[os.path.join(env['DESTDIRPREFIX'],"lib")])
elif "win" in os.sys.platform:
    env.Alias("install",[os.path.join(env['PREFIX'],"Lib")])

#add aliases for documentation target
env.Alias("doc",[os.path.join("doc","manual","xrutils.pdf")])

debug = ARGUMENTS.get('debug', 0)
if int(debug):
    env.Append(CCFLAGS=["-g","-O0"])
else:
    env.Append(CCFLAGS=["-O2"])

############################
#   installation related
#  installs python package
############################

if "install" in COMMAND_LINE_TARGETS:
    #write the clib_path.conf file
    conffilename = os.path.join(".","python","xrutils","clib_path.conf")
    fid = open(conffilename,"w")
    if os.sys.platform == "darwin":
        libpath = os.path.join(env['PREFIX'],"lib","libxrutils.dylib")
    elif os.sys.platform == "linux2":
        libpath = os.path.join(env['PREFIX'],"lib","libxrutils.so")
    elif "win" in os.sys.platform:
        libpath = os.path.join(env['PREFIX'],"Lib","xrutils.dll")
    fid.write("[xrutils]\n")
    fid.write("clib_path = %s\n" %libpath)
    fid.close()
    print("create clib_path.conf file (libfile: %s)"%(libpath))
    #run python installer
    if "win" in os.sys.platform or env['DESTDIR'] == "" or env['DESTDIR'] == "${DESTDIR}":
        python_installer = subprocess.Popen("python setup.py install --prefix=%s"%(env['PREFIX']),shell=True)
    else:
        python_installer = subprocess.Popen("python setup.py install --root=%s --prefix=%s" %(env['DESTDIR'],env['PREFIX']),shell=True)
    python_installer.wait()

############################
#    config like things
############################

def CheckPKGConfig(context, version):
    context.Message( 'Checking for pkg-config... ' )
    ret = context.TryAction('pkg-config --atleast-pkgconfig-version=%s' % version)[0]
    context.Result( ret )
    return ret

def CheckPKG(context, name):
    context.Message( 'Checking for %s... ' % name )
    ret = context.TryAction('pkg-config --exists \'%s\'' % name)[0]
    context.Result( ret )
    return ret

# check for headers, libraries and packages
if not env.GetOption('clean') or not env.GetOption('help'):

    conf = Configure(env,custom_tests = { 'CheckPKGConfig' : CheckPKGConfig, 'CheckPKG' : CheckPKG })
    if not conf.CheckCC():
        print('Your compiler and/or environment is not correctly configured.')
        Exit(1)
    
    if not conf.CheckHeader(['stdlib.h','stdio.h','math.h','time.h']):
        print 'Error: did not find one of the needed headers!'
        Exit(1)
   
    if not conf.CheckLibWithHeader('gomp','omp.h','c'):
        print 'Warning: did not find openmp + header files -> using serial code'
    else:
        env.Append(CCFLAGS=['-fopenmp','-D__OPENMP__'],LIBS=['gomp'])

    if not conf.CheckLibWithHeader('pthread','pthread.h','c'):
        print 'Error: did not find pthread + header files!'
    else:
        env.Append(LIBS=['pthread'])

    if not conf.CheckLib(['m']):
        print 'Warning: did not find one of the needed libraries!'
        if "win" not in os.sys.platform:
            Exit(1)

    conf.Finish()

############################
# tarball creation/packaging 
############################

# package xrayutilities into a tarball for distribution
#print("Creating tarball for redistribution of xrayutilities...")
env['DISTTAR_FORMAT']='gz'
env.Append(
    DISTTAR_EXCLUDEEXTS=['.o','.os','.so','.a','.dll','.dylib','.cache','.dblite','.pyc','.log','.out','.aux','.fls','.toc'], 
    DISTTAR_EXCLUDEDIRS=['.git','.sconf_temp', 'dist', 'build'],
    DISTTAR_EXCLUDERES=[r'clib_path.conf','.gitignore'])

env.DistTar(os.path.join("dist","xrayutilities_"+datetime.date.today().isoformat()), [env.Dir(".")]) 


############################
#  include sub-directories
############################

Export("env")

#add subdirectories
SConscript(["src/SConscript","doc/manual/SConscript"])
