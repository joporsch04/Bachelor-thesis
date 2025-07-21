# Bachelor Thesis
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16223179.svg)](https://doi.org/10.5281/zenodo.16223179)

## Author
Johannes Porsch (Jo.Porsch@campus.lmu.de, joporsch04@gmail.com)

## Abstract
Understanding the temporal evolution of strong-field ionization, often described through an ionization rate, is fundamental for controlling and interpreting electron motion on its natural, attosecond timescale.
While the Strong Field Approximation (SFA) is a common theoretical tool for modeling this process, standard SFA models fail to accurately reproduce certain ionization dynamics and predict ionization yields that differ by orders of magnitude from numerical solutions of the time-dependent Schrödinger equation (TDSE).

To address these shortcomings, this work develops an extended SFA formalism that incorporates pre-ionization dynamics, including the Stark effect, ground state distortion, and transitions to excited states. This is achieved by using the time-dependent coefficients to enhance the rate, which are determined by solving the TDSE both within a restricted subspace of bound states and in the full Hilbert space.

The results demonstrate that the extended SFA model, using coefficients from the full Hilbert space simulation, significantly improves the reconstruction of off-cycle ionization dynamics compared to standard SFA. The improvement is almost entirely due to the phase evolution of the ground-state coefficient, with its amplitude having only a negligible effect. Surprisingly, the conventional AC Stark shift, isolated using the subspace-restricted coefficients, was shown to play only a minor role. This suggests that the crucial phase contribution arises from effects that are only captured when ionization is allowed in the simulation.

While the model successfully improves reconstructing the ionization dynamics, the large discrepancy in the absolute ionization yield remains. Preliminary findings suggest that including transitions to excited states within the extended SFA framework could partially bridge this gap, marking a key direction for future research.

## Key Publications
- [Anatomy of strong field ionization](https://doi.org/10.1080/0950034042000275360)
- [Theory of Nonlinear Photoconductive Sampling in Atomic Gases](https://doi.org/10.1002/andp.202300322)
- [An environment for solving schrödinger-like problems](https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX)
- [A general approximator for strong-field ionization rates](https://arxiv.org/abs/2507.03996)

## Version Information
The official submission of this thesis corresponds to version **v1.0.0**. However, the repository will be updated continuously with future improvements and additions.

## Git
```
git clone https://github.com/joporsch04/Bachelor-thesis
```
The folder TIPTOE-Hydrogen was created during a research assistant position at MPQ during summer 2024 and not part of this thesis. It is a git submodule and can be cloned with:
```
git submodule update --init --recursive
cd TIPTOE-Hydrogen
git pull origin main
```

## tRecX
For the installation of tRecX, please visit this [link](https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX). For simple testing I recommend using [Docker](https://www.docker.com/). The `dockerfile` can be found in the folder tRecX:
[link](https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX)
```
cd tRecX
docker build -t trecx .
docker run -it trecx
```
If you choose to use a VM, pay attention to the Linux version. In some distributions libatlas-base-dev and libboost-all-dev are not compatible. In any case, the `dockerfile` (Ubuntu 22.04) works fine. The changes I made in tRecX are in the folder tRecX. The `dockerfile` already has the changes implemented. 

## Python
The ionisation model is implemented in python. 

## Max Planck Institute of Quantum Optics (MPQ)
For more information about the institute, please visit the official website:
[MPQ Homepage](https://www.mpq.mpg.de)
