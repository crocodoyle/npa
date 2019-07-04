from setuptools import setup

setup(
    name='npa',
    version='0.1',
    packages=['npa'],
    url='https://github.com/crocodoyle/npa',
    license='Apache License, 2.0',
    author='Andrew Doyle',
    author_email='andrew.doyle@mcgill.ca',
    description='Neural Power Amplifier',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=['neuroscience', 'neural oscillations', 'power spectra', '1/f', 'electrophysiology'],
    install_requires=['numpy', 'scipy>=1.2', 'fooof', 'mne'],
)
