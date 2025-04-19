# Bachelor Thesis

## Author
Johannes Porsch (Jo.Porsch@campus.lmu.de, joporsch04@gmail.com)

## Abstract
Multiphoton ionization of atoms in strong laser fields is a fundamental process in attosecond physics. In this work, we extend the strong-field approximation (SFA) by incorporating the influence of excited atomic states on ionization rates. Standard SFA formulations neglect these excited states, assuming that the laser field has no effect on the atom before ionization. However, in intense few-cycle laser pulses, the Stark shift and transient population of excited states can significantly modify ionization dynamics. We numerically solve the time-dependent Schrödinger equation (TDSE) using the tRecX code to extract time-dependent probability amplitudes for hydrogen’s ground and excited states. These amplitudes are then integrated into the SFA formalism to evaluate their impact on ionization rates. 

## Key Publications
- [Anatomy of strong field ionization](https://doi.org/10.1080/0950034042000275360)
- [Direct sampling of a light wave in air](https://doi.org/10.1364/OPTICA.5.000402)
- [An environment for solving schrödinger-like problems](https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX)

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


## Max Planck Institute of Quantum Optics (MPQ)
For more information about the institute, please visit the official website:
[MPQ Homepage](https://www.mpq.mpg.de)
