project('xrayutilities', 'c', version : '1.7.11')

use_openmp = get_option('use_openmp')

# Fetch the C compiler
cc = meson.get_compiler('c')

# Import Python module
pyx = import('python').find_installation(pure: false)

# Fetch Numpy include directory
numpy_include = run_command(pyx, '-c', 'import numpy; print(numpy.get_include())', check: true).stdout().strip()

# Include paths
include_dirs = include_directories('src', numpy_include)

# Find OpenMP dependency
openmp_dep = dependency('openmp', required : use_openmp)

# Add the Python extension module
pyx.extension_module(
  'cxrayutilities',  # Python extension name
  sources : files(
    'src/cxrayutilities.c',
    'src/array_utils.c',
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
  subdir: 'xrayutilities',
  dependencies: openmp_dep,
)

# Install the Python extension
install_subdir('lib/xrayutilities', install_dir : pyx.get_install_dir())

db_build = custom_target(
  'build_database',
  env: ['HDF5_USE_FILE_LOCKING=FALSE'],
  input : 'lib/xrayutilities/materials/_create_database.py',
  output : 'elements.db',
  command : [pyx, '@INPUT@', '@OUTPUT@'],
  install: true,
  install_dir: join_paths(pyx.get_install_dir(), 'xrayutilities/materials/data'),
)
