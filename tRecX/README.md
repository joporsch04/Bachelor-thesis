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