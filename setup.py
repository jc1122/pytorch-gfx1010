from setuptools import find_packages, setup


setup(
    py_modules=["pytorch_gfx1010_autoload"],
    packages=find_packages(include=["workarounds*"]),
)
