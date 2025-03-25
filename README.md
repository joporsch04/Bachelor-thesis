# Bachelor Thesis

## Author
Johannes Porsch (Jo.Porsch@campus.lmu.de, joporsch04@gmail.com)

## Abstract
Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.

## Key Publications
- [Direct sampling of a light wave in air](https://doi.org/10.1364/OPTICA.5.000402)
- [An environment for solving schrödinger-like problems](https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX)

## Git
```
git clone https://github.com/joporsch04/Bachelor-thesis
```
<!-- The folder TIPTOE-Hydrogen was created during a research assistant position at MPQ during summer 2024 and not part of this thesis. It is a git submodule and can be cloned with: -->
```
git submodule update --init --recursive
cd TIPTOE-Hydrogen
git pull origin main
```

## tRecX
<!-- For the installation of tRecX, please visit this [link](https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX). For simple testing I recommend using [Docker](https://www.docker.com/). The `dockerfile` can be found in the folder tRecX: -->
[link](https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX)
```
cd tRecX
docker build -t trecx .
docker run -it trecx
```
<!-- If you choose to use a VM, pay attention to the Linux version. In some distributions libatlas-base-dev and libboost-all-dev are not compatible. In any case, the `dockerfile` (Ubuntu 22.04) works fine. 

## Max Planck Institute of Quantum Optics (MPQ)
For more information about the institute, please visit the official website:
[MPQ Homepage](https://www.mpq.mpg.de) -->
