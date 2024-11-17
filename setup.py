from setuptools import find_packages, setup

setup(
    name="fx_ir",
    description="An unambiguous dialect of PyTorch's FX IR.",
    version="0.1.0",
    author="Matteo Spallanzani",
    author_email="matteospallanzani@hotmail.it",
    license="",
    url="",
    package_dir={"": "src"},
    packages=find_packages(where="src", include="fx_ir"),
    install_requires=["torch"],
)
