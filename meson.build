project('xrayutilities', 'c',
  version : run_command('python', '-c',
    '''
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open("pyproject.toml", "rb") as f:
        print(tomllib.load(f)["project"]["version"])
    '''
  ).stdout().strip()
)

enable_openmp = get_option('enable_openmp')

# Fetch the C compiler
cc = meson.get_compiler('c')

# Import Python module
pyx = import('python').find_installation()

# Fetch Numpy include directory
numpy_include = run_command(pyx, '-c', 'import numpy; print(numpy.get_include())', check: true).stdout().strip()

# Include paths
include_dirs = include_directories('src', numpy_include)

# Find OpenMP dependency
openmp_dep = dependency('openmp', required : enable_openmp)

# Handle OpenMP support based on the option and dependency status
if enable_openmp
  if not openmp_dep.found()
    error('OpenMP was explicitly enabled but could not be found on this system.')
  else
    message('OpenMP support enabled')
    # Add compiler-specific OpenMP flags
    if cc.get_id() == 'gcc' or cc.get_id() == 'clang' or cc.get_id() == 'mingw'
      add_project_arguments('-fopenmp', language: 'c')
      add_project_link_arguments('-lgomp', language: 'c')
    elif cc.get_id() == 'msvc'
      add_project_arguments('/openmp', language: 'c')
    endif
  endif
else
  message('OpenMP support disabled')
endif

# Add the Python extension module
pyx.extension_module(
  'cxrayutilities',  # Python extension name
  sources : files(
    'src/cxrayutilities.c',
    'src/block_average.c',
    'src/file_io.c',
    'src/gridder1d.c',
    'src/gridder2d.c',
    'src/gridder3d.c',
    'src/gridder_utils.c',
    'src/hklcond.c',
    'src/qconversion.c'
  ),
  include_directories : include_dirs,
  install: true,
  install_dir: join_paths(pyx.get_install_dir(), 'xrayutilities'),
)

# Install the Python extension
install_subdir('lib/xrayutilities', install_dir : pyx.get_install_dir())

db_build = custom_target(
  'build_database',
  input : 'lib/xrayutilities/materials/_create_database.py',
  output : 'elements.db',
  command : [pyx, '@INPUT@', '@OUTPUT@'],
  install: true,
  install_dir: join_paths(pyx.get_install_dir(), 'xrayutilities/materials/data'),
)
