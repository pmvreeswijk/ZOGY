from setuptools import setup, find_packages
setup(
    name='zogy',
    version='1.3.0',
    description='a Python implementation of proper image subtraction (ZOGY 2016, ApJ, 830, 27)',
    url='https://github.com/pmvreeswijk/ZOGY',
    author='Paul Vreeswijk, Kerry Paterson',
    author_email='pmvreeswijk@gmail.com',
    python_requires='>=3.6',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'astropy', 'matplotlib', 'scipy',
                      'pyfftw', 'lmfit', 'sip_tpv', 'scikit-image', 'healpy',
                      'photutils', 'tensorflow', 'fitsio', 'fsspec', 'gcsfs']
)
