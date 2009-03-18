import os

AddOption("--prefix",dest="prefix",type="string",
          default="/usr/local",metavar="INSTPREFIX",
          action="store",nargs=1)

env = Environment(PREFIX=GetOption("prefix"),
                  CCFLAGS=["-fPIC","-Wall","-pthread"],
                  LIBS=["m","pthread"])

if "install" in COMMAND_LINE_TARGETS:
    #write the config.py file
    fid = open("./python/xrutils/config.py","w")
    pref = GetOption("prefix")
    libpath = os.path.join(pref,"lib/libxrayutils.so")
    fid.write("clib_path = \"%s\"" %libpath)
    fid.close()

#add the aliases for install target
ilpath = env.Alias("instlib","$PREFIX/lib")
iltool = env.Alias("insttool","$PREFIX/bin")
env.Alias("install",[ilpath,iltool])

#add aliases for documentation target
ugdoc = env.Alias("ugdoc","doc/manual/xrutils.pdf")
env.Alias("doc",[ugdoc])

dbg = env.Clone()
opt = env.Clone()

dbg.Append(CCFLAGS=["-g","-O0"])
opt.Append(CCFLAGS=["-O2"])
Export("dbg")
Export("opt")

#add subdirectories
SConscript(["src/SConscript","doc/manual/SConscript"])
