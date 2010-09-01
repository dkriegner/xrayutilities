import os

AddOption("--prefix",dest="prefix",type="string",
          default="/usr/local",metavar="INSTPREFIX",
          action="store",nargs=1)

env = Environment(PREFIX=GetOption("prefix"),ENV=os.environ,
                  CCFLAGS=["-fPIC","-Wall","-std=c99"])
                  
                  #CCFLAGS=["-fPIC","-Wall","-pthread"],
                  #LIBS=["m","pthread"])

if "install" in COMMAND_LINE_TARGETS:
    #write the config.py file
    conffilename = os.path.join(".","python","xrutils","config.py")
    fid = open(conffilename,"w")
    pref = GetOption("prefix")
    if os.sys.platform == "darwin":
        libpath = os.path.join(pref,"lib","libxrutils.dylib")
    elif os.sys.platform == "linux2":
        libpath = os.path.join(pref,"lib","libxrutils.so")
    elif "win" in os.sys.platform:
        libpath = os.path.join(pref,"lib","xrutils.dll")
    fid.write("clib_path = r\"%s\"" %libpath)
    fid.close()

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
if not env.GetOption('clean'):

    conf = Configure(env,custom_tests = { 'CheckPKGConfig' : CheckPKGConfig, 'CheckPKG' : CheckPKG })
    if not conf.CheckCC():
        print('Your compiler and/or environment is not correctly configured.')
        Exit(1)
    
    #if not conf.CheckPKGConfig('0.20.0'):
    #    print 'pkg-config >= 0.20.0 not found.'
    #    Exit(1)
 
    #if not conf.CheckPKG('cblas'):
    #    print 'cblas not found.'
    #    Exit(1)

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
        env.Append(CCFLAGS=['-pthread'],LIBS=['pthread'])

    if not conf.CheckLib(['m']):
        print 'Error: did not find one of the needed libraries!'
        Exit(1)

    conf.Finish()

#env.ParseConfig('pkg-config --cflags --libs cblas')

#add the aliases for install target
env.Alias("install",["$PREFIX/lib"])#,"$PREFIX/bin"])

#add aliases for documentation target
env.Alias("doc",["doc/manual/xrutils.pdf"])

debug = ARGUMENTS.get('debug', 0)
if int(debug):
    env.Append(CCFLAGS=["-g","-O0"])
else:
    env.Append(CCFLAGS=["-O2"])

Export("env")

#add subdirectories
SConscript(["src/SConscript","doc/manual/SConscript"])
