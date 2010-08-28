from distutils.core import setup

setup(name="xrutils",
      version="0.5",
      author="Eugen Wintersberger",
      description="package for x-ray diffraction data evaluation",
      author_email="eugen.wintersberger@jku.at",
      maintainer="Eugen Wintersberger",
      maintainer_email="eugen.wintersberger@jku.at",
      package_dir={'':'python'},
      packages=["xrutils","xrutils.math","xrutils.vis",
                "xrutils.io","xrutils.materials"],
      package_data={"xrutils.materials":["data/*.db"]}
      )
