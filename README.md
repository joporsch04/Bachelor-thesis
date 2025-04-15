# Bachelor Thesis

## Author
Johannes Porsch (Jo.Porsch@campus.lmu.de, joporsch04@gmail.com)

## Abstract
Multiphoton ionization of atoms in strong laser fields is a fundamental process in attosecond physics. In this work, we extend the strong-field approximation (SFA) by incorporating the influence of excited atomic states on ionization rates. Standard SFA formulations neglect these excited states, assuming that the laser field has no effect on the atom before ionization. However, in intense few-cycle laser pulses, the Stark shift and transient population of excited states can significantly modify ionization dynamics. We numerically solve the time-dependent Schrödinger equation (TDSE) using the tRecX code to extract time-dependent probability amplitudes for hydrogen’s ground and excited states. These amplitudes are then integrated into the SFA formalism to evaluate their impact on ionization rates. 

## Key Publications
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
If you choose to use a VM, pay attention to the Linux version. In some distributions libatlas-base-dev and libboost-all-dev are not compatible. In any case, the `dockerfile` (Ubuntu 22.04) works fine. 

## Changes in tRecX
in run_trecx.cpp line 898
```
// Ändere Duals zu Smart Pointern
std::vector<std::shared_ptr<Coefficients>> Duals; // Oder unique_ptr

if (printOps.size() > 1) {
    std::cout << "now at calc eigenvecs" << std::endl;
    std::vector<int> states = {0, 1, 2, 3, 4, 5};
    std::cout << "printOps[0]=" << printOps[0]->name() << std::endl;
    const OperatorAbstract* specOp(printOps[1]);

    int kend = *std::max_element(states.begin(), states.end()) + 1;

    // slv bleibt lokal, da wir die Daten kopieren
    EigenSolver slv(-DBL_MAX, DBL_MAX, kend, true, true, false, "Lapack");
    slv.withSelect("SmallReal[8]");
    slv.fullVectors().compute(specOp, printOps[0]);
    slv.select("SmallReal[8]");
    slv.orthonormalize();

    // Hole die (wahrscheinlich) nicht-besitzenden rohen Zeiger vom Solver
    std::vector<Coefficients*> rawDuals = slv.dualVectors();

    // Erstelle Kopien und speichere shared_ptr auf die Kopien
    Duals.reserve(rawDuals.size());
    for (const Coefficients* rawPtr : rawDuals) {
        if (rawPtr) {
            // Erzeuge eine Kopie auf dem Heap und verwalte sie mit shared_ptr
            Duals.push_back(std::make_shared<Coefficients>(*rawPtr));
            // ^-- Benötigt einen funktionierenden Kopierkonstruktor für Coefficients!
        } else {
            Duals.push_back(nullptr); // Behalte nullptr bei, falls vorhanden
        }
    }
    // Wenn slv jetzt zerstört wird, löscht es die Originale.
    // Deine Kopien in Duals bleiben aber gültig.

} else {
    std::cout << "calculating eigenvecs was not possible, set duals empty" << std::endl;
    Duals.clear(); // Leert den Vektor von shared_ptrs
}

// Verwende nun den 'Duals'-Vektor von shared_ptrs im weiteren Code.
// Du musst den Code in eigenProjection anpassen, um shared_ptr zu akzeptieren
// oder .get() zu verwenden, um rohe Zeiger zu erhalten.
```
in timePropagatorOutput.cpp 
```
static std::complex<double> eigenProjection(int IOp, std::vector<OperatorAbstract*> Ops, double Time, const Coefficients* Wf,bool Normalize, std::vector<std::shared_ptr<Coefficients>> Duals){
    Ops[IOp]->update(Time,Wf);
    if (IOp > 1 && !Duals.empty() && IOp - 2 < Duals.size()) {
        const Coefficients* constdual_raw = Duals[IOp - 2].get();
        complex<double> expec=Ops[IOp]->matrixElementUnscaled(*constdual_raw,*Wf);
        expec=Threads::sum(expec);
        if(Normalize){
            std::complex<double> nrm=1.;
            if(Ops[IOp]->name().find("Ovr(")==std::string::npos)){
                for(auto o: Ops){
                    if(o->name().find("Ovr(")!=std::string::npos)){
                        nrm=Threads::sum(o->matrixElementUnscaled(*constdual_raw,*Wf));
                        break;
                    }
                }
            }
            if(nrm!=0.)expec/=nrm;
        }
        return expec;
    }
    else {
        complex<double> expec=Ops[IOp]->matrixElementUnscaled(*Wf,*Wf);
        expec=Threads::sum(expec);
        if(Normalize){
            std::complex<double> nrm=1.;
            if(Ops[IOp]->name().find("Ovr(")==std::string::npos)){
                for(auto o: Ops){
                    if(o->name().find("Ovr(")!=std::string::npos)){
                        nrm=Threads::sum(o->matrixElementUnscaled(*Wf,*Wf));
                        break;
                    }
                }
            }
            if(nrm!=0.)expec/=nrm;
        }
        return expec;
    }
}
```

## Debugging
make checkpoints
```
cd tRecX
cmake -DCMAKE BUILD TYPE=Develop . // or cmake -DCMAKE_BUILD_TYPE=Develop -DCMAKE_CXX_FLAGS_DEVELOP="-g -O0" .
make -j6
gdb
set args tiptoe.inp
run
backtrace
frame 1 // depends wich checkpoint you want to debug
```

## Max Planck Institute of Quantum Optics (MPQ)
For more information about the institute, please visit the official website:
[MPQ Homepage](https://www.mpq.mpg.de)
