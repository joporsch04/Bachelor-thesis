# Bachelor Thesis

## Author
Johannes Porsch (Jo.Porsch@campus.lmu.de, joporsch04@gmail.com)

## Abstract
direct sampling of light wave in air -> TIPTOE

## Key Publications
- [Direct sampling of a light wave in air](https://doi.org/10.1364/OPTICA.5.000402)
- [An environment for solving schrödinger-like problems](https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX)

## Git
```
git clone https://github.com/joporsch04/Bachelor-thesis
```
The folder TIPTOE was created during a research assistant position at MPQ during summer 2024. It is a git submodule and can be cloned with:
```
git submodule update --init --recursive
```

## tRecX
For the installation of tRecX, please visit this [link](https://gitlab.physik.uni-muenchen.de/AG-Scrinzi/tRecX). For simple testing I recommend using [Docker](https://www.docker.com/). The `dockerfile` can be found in the folder tRecX:
```
cd tRecX
docker build -t trecx .
docker run -it trecx
```

## Max Planck Institute of Quantum Optics (MPQ)
For more information about the institute, please visit the official website:
[MPQ Homepage](https://www.mpq.mpg.de)