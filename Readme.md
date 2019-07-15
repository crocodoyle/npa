# NPA - Neural Power Amplifier

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![CircleCI](https://circleci.com/gh/crocodoyle/npa.svg?style=svg)](https://circleci.com/gh/crocodoyle/npa)

NPA leverages the [FOOOF](https://github.com/fooof-tools/fooof/) parametric model of neural power spectra to select, amplify, or attenuate different components of neural time series to compare decoding experiments and investigate the contributions of each component to brain function.

Example usage: https://nbviewer.jupyter.org/github/crocodoyle/npa/blob/master/npa/NPA%20Example.ipynb

## Reference

If you use this code in your project, please cite:

    Doyle, JA, Toussaint, PJ, Evans, AC. (2019) Amplifying the Neural Power Spectrum. bioRxiv, 659268.
    doi: https://doi.org/10.1101/659268

Link: https://www.biorxiv.org/content/10.1101/659268v1.abstract

## Dependencies

NPA is written in Python, and requires Python >= 3.5 to run. It has the following dependencies:
- numpy
- scipy >= 1.2
- fooof
- mne >= 0.17