from setuptools import setup, find_packages

setup(
    name='meli',
    description='A package for applying to MercadoLibre.',
    packages=find_packages(where='src'),
    package_dir={'':'src'},
)