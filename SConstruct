import os

AddOption("--prefix",dest="prefix",type="string",
          default="/usr/local",metavar="INSTPREFIX",
          action="store",nargs=1)

env = Environment(PREFIX=GetOption("prefix"),ENV=os.environ,
                  CCFLAGS=["-fPIC","-Wall","-pthread"],
                  LIBS=["m","pthread"])

if "install" in COMMAND_LINE_TARGETS:
    #write the config.py file
    fid = open("./python/xrutils/config.py","w")
    pref = GetOption("prefix")
    libpath = os.path.join(pref,"lib/libxrutils.so")
    fid.write("clib_path = \"%s\"" %libpath)
    fid.close()

#add the aliases for install target
env.Alias("install",["$PREFIX/lib","$PREFIX/bin"])

#add aliases for documentation target
env.Alias("doc",["doc/manual/xrutils.pdf"])

dbg = env.Clone()
opt = env.Clone()

dbg.Append(CCFLAGS=["-g","-O0"])
opt.Append(CCFLAGS=["-O2"])
Export("dbg")
Export("opt")

#add subdirectories
SConscript(["src/SConscript","doc/manual/SConscript"])
