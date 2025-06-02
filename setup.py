from setuptools import setup, find_packages

setup(
    name='IHSetMOOSE',
    version='0.0.5',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'pandas',        
        'numba',
        'matplotlib',
        'IHSetUtils @ git+https://github.com/IHCantabria/IHSetUtils.git'
    ],
    author='Lucas de Freitas',
    author_email='lucas.defreitas@unican.es',
    description='IH-SET Jaramillo et al. (2022)',
    url='https://github.com/IHCantabria/IHSetIH-MOOSE',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)