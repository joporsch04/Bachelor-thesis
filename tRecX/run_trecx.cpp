// tRecX = tSurff+irECS - a universal Schroedinger solver

// tRecX = tSurff+irECS - a universal Schroedinger solver
// Copyright (c) 2015 - 2024 by Armin Scrinzi (armin.scrinzi@lmu.de)
// 
// This program is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free Software Foundation; 
// either version 2 of the License, or (at your option) any later version.
// End of license

#define EIGEN_MATRIXBASE_PLUGIN "EigenAddonMatrix.h"

#include "operatorHaCC.h"

#include "indexOverlap.h"
#include "tree.h"

#include "timer.h"
#include "readInput.h"
#include "units.h"
#include "tRecX_cache.h"
#include "printOutput.h"
#include "mpiWrapper.h"
#include "debugInfo.h"
#include "threads.h"

#include "discretizationtsurffspectra.h"
#include "discretizationSurface.h"
#include "discretizationConstrained.h"

#include "pulse.h"
#include "harmonics.h"
#include "plot.h"
#include "scanEigenvalue.h"
#include "eigenSubspace.h"
#include "floquetAnalysis.h"
#include "potentialTruncate.h"
#include "operatorDefinition.h"
#include "operatorDefinition.h"
#include "operatorFactor.h"
#include "operatorFloor3d.h"
#include "operatorFloorEE.h"
#include "parallelLayout.h"

#include "timePropagator.h"
#include "timePropagatorOutput.h"
#include "derivativeFlatInhomogeneous.h"

#include "tsurffSource.h"
#include "derivativeLocal.h"
#include "operatormapchannelssurface.h"
#include "operatorGradient.h"
#include "operatorIdentity.h"
#include "autoconverge.h"
#include "spectrumISurff.h"

#include "odeFactory.h"

#include "tRecXchecks.h"
#include "channelsSubregion.h"

#include "densityOfStates.h"

#include "initialState.h"

#include "coefficientsWriter.h"
#include "log.h"

#include "basisOrbital.h"
#include "basisOrbitalNumerical.h"
#include "productFunction.h"

#include "spectrumPlot.h"
#include "dipoleBoundContinuum.h"
#include "staticIonization.h"

#ifdef _USE_UI_
#include "uiLayout.h"
#endif

#include "eigenTrace.h"

#include "hamiltonianIon.h"
#include "propagateChannels.h"

// develop only
#include "surfaceFlux.h"
#include "indexConstraint.h"
#include "indexNew.h"
#include "multipolePotential.h"
#include "randomPotential.h"
#include "timeCritical.h"
#include "farm.h"

#include "qcInput.h"

using namespace std;
using namespace tools;

TIMER(all, )
TIMER(run, )
TIMER(setup, )
TIMER(buildOperators, )
TIMER(matrix, )
TIMER(setupDisc, )
TIMER(setupOrbital, )
TIMER(setupPLot, )
TIMER(setupSpectral, )
TIMER(input, )
TIMER(setProp, )
TIMER(setProp1, )
TIMER(setProp2, )
TIMER(setProp3, )
TIMER(setProp4, )
TIMER(initial,)
TIMER(other, )
TIMER(timePropagate,)

// auxiliary sections for main code
#include "tRecX.h"
#include <string>
#include "version.h"
static bool generateLinp()
{
    return ReadInput::main.flag("generate-linp", "only generate list-of-inputs file");
}

// a single "run": use single input file
void tRecX::run_trecx(std::string InputFile, int argc, char *argv[])
{
    //    Sstr+"try this A"+boost::math::sph_bessel(1,0.27557017407092*20)+Sendl;

    tRecX::initialize(); // clear possible settings from preceding call

    Threads::setup(MPIwrapper::communicator());

    START(all);
    LOG_ON();

    //============================================================================================
    // input and part of setup
    //==============================================================================================
    // only overall master outputs to screen
    if (not MPIwrapper::isMaster(MPIwrapper::worldCommunicator()))
        PrintOutput::off("Screen");

    // only master of current flock outputs to file
    if (not MPIwrapper::isMaster(MPIwrapper::communicator()))
        PrintOutput::off("Both");

    ReadInput::openMain(InputFile, argc, argv);


    ReadInput::main.obsolete("Channel",ReadInput::anyName,"Channel input category has been removed");

    Farm farm(ReadInput::main); // make sure possible inputs to Farm are known

    // need to reset


    debug_tools::setVerboseLevel();
    PrintOutput::subTitle("-> tRecX (v 2.0) (C) 2015-2024 Armin Scrinzi (licenced under GPLv2) <-");

    std::string build("Parallel");
#ifdef _NOMPI_
    build+="Scalar";
#endif
#ifdef _DEVELOP_
    build+="Develop";
#endif
    PrintOutput::paragraph();
    PrintOutput::subTitle("Build: "+std::string(__DATE__)+" "+__TIME__+", git hash "+gitHashAtCmake+", type \""+build+"\"");
#ifdef _DEVELOP_
    PrintOutput::subTitle("Git: "+gitLogAtCmake);
#endif

    tRecX::tutorialMessage(ReadInput::main.output());
    PrintOutput::paragraph();

    // open output streams for calculation
    if(MPIwrapper::isMaster()){
        PrintOutput::set(ReadInput::main.outputTopDir()+PrintOutput::outExtension);
    }

    string title;
    ReadInput::main.read("Title","",title,"tRecX calculation","title that will be printed to output");

    PrintOutput::title("Date ["+Timer::currentDateTime()+"] "+title);

    PrintOutput::paragraph();

    Timer::monitorStart(ReadInput::main.outputTopDir(),ReadInput::main);
    Timer::monitor("setup");

    Units::setDefault("au");        // general default units
    ReadInput::main.setUnits("au"); // convert input to these units

    ProductFunction::read(ReadInput::main);
    ParallelLayout::read(ReadInput::main);

#ifdef _DEVELOP_
    //***************************************************************
    // for parallel debuging
    // for starting two xterms where both mpi processes run under debugger - useful to find out where
    // exactly a processes gets stuck or crashes
    //
    // >mpirun -n 2 xterm -hold -e gdb -ex run --args ./tRecX 26haCCdesign
    //
    // otherwise:
    // compile with -O0
    // mpirun -np 4 ...
    // ps aux | grep tRecX
    // gdb tRecX procNumber
    // gdb> set DebugWait=0 to start each process
    if (ReadInput::main.flag("DEBUGparallel", "block, see main_trecx.cpp how to proceed"))
    {
        unsigned int DebugWait = 1;
        while (DebugWait)
            ;
        MPIwrapper::Barrier();
    }
#endif
    //***************************************************************

#ifdef _OPENMP
    PrintOutput::title("Open MP");
#endif

    tRecX_cache::read(ReadInput::main);

    tRecX::readConstantsAndParameters();

    OperatorDefinition::setup();

    tRecX::readBases(ReadInput::main);
    RandomPotential::read(ReadInput::main);
    Autoconverge autcon(ReadInput::main);

    // command line flags
    bool showOperators = ReadInput::main.flag("showOperators", "show operator structure and stop");
    std::string printMatrices;
    ReadInput::main.read("Flag", "printMatrices", printMatrices, "",
                         "show operator matrix, "
                         "norms[depth,digits]...norms of matrix-blocks to depth,"
                         "full[digits]...all, "
                         "block[ibl,jbl,i0,j0]...block of full matrix",
                         0, "printMatrices");

    // read possible multi-dimensional bases

    string halfPlane = "";
#ifdef _DEVELOP_
    double hpR, hpS, hpEta;
    ReadInput::main.read("_EXPERT_", "halfPlaneR", hpR, "0", "smoothing for half-plane suppression");
    ReadInput::main.read("_EXPERT_", "halfPlaneS", hpS, "5", "smoothing for half-plane suppression");
    ReadInput::main.read("_EXPERT_", "halfPlaneEta", hpEta, "0.2", "smoothing for half-plane suppression");
    if (hpR > 0.)
    {
        halfPlane = "<1><1-trunc[0," + tools::str(hpEta) + "]><1-trunc[" + tools::str(hpR - hpS) + "," + tools::str(hpR) + "]>";
    }
#endif
    int hugeMatrix;
    ReadInput::main.read("Flag","hugeMatrix",hugeMatrix,"3000","set limit where matrix will be considered huge",1,"hugeMatrix")
        .texdocu(R"tex(
                     The function OperatorAbstract::isHuge() returns true if matrix dimension exeed this value.
                     The function is used in different contexts and may either cause an error stop or the use of approximate
                     algorithms. Large values can cause excessive use of memory and/or compute time. Change only if needed.
                     )tex");


#ifdef _USE_UI_
    UiLayout::setPage("System");
#endif
    Pulse::read(ReadInput::main, true);
#ifdef _USE_UI_
    UiLayout::setPage();
#endif
    // supplement input with lines from default inputs
    QcInput::axes(ReadInput::main);

    OperatorFactor::readMatrix(ReadInput::main);
    ChannelsSubregion::read(ReadInput::main);
    OperatorFloorEE::read(ReadInput::main);
    OperatorFloor3d::read(ReadInput::main);
    tRecX::PlotFunctions(ReadInput::main);
    BasisOrbitalNumerical::read(ReadInput::main);
    IndexNew::read(ReadInput::main);
    SpectrumISurff iSurff(ReadInput::main);

    tRecX::readExtra(ReadInput::main);

    // Eigenproblem xor initial value problem
    ReadInput::main.exclude("Initial", "Eigen");

    string eigenSelect;
    bool eigenVectors, eigenElements;
    ReadInput::main.read("Eigen", "select", eigenSelect, "NONE",
                         "kinds: All, SmallReal[N]...first N, LargeAbs[N]...last N, Nearest[N,Ereal,Eimag]...in complex plane, Rectangle[Rmin,Rmax,Imin,Imax]...in complex plane, NONE",
                         1, "eigenSelect")
        .texdocu(R"tex(
                     \begin{itemize}
                     \item[All] only meaningfull with Lapack for \nameref{docu:Eigen:method}
                     \item[SmallReal] sort by real part, return lowest N
                     \item[LargeAbs] sort by modulus, return highest N
                     \item[Nearest] locate N complex $E_i$ with smallest $|E_i-$Ereal-i Eimag$|$ (best wit Arpack, inverse iteration)
                     \item[Recangle] locate all within rectangle of complex plane
                     \end{itemize}
                     )tex");
    ReadInput::main.read("Eigen", "vectors", eigenVectors, "false", "save eigenvectors (binary format)", 1, "eigenVectors");
    ReadInput::main.read("Eigen", "elements", eigenElements, "false", "compute matrix elements of Operator expectation ", 1, "eigenElements");

    string initialKind;
    int initialN;
    ReadInput::main.read("Initial", "state", initialN, "0", "initial state (number of state in Hamiltonian)");
    initialKind = InitialState::readKind(ReadInput::main);

#ifdef _USE_UI_
    UiLayout::setPage("System");
#endif
    std::string hamDef = tRecX::readOperator("hamiltonian");
    std::string intDef = tRecX::readOperator("interaction");
    std::string iniDef = tRecX::readOperator("initial");
    std::string preconDef = tRecX::readOperator("projection");
    std::string paramDef = tRecX::readOperator("parameterTerm");
    std::string expecDef = readOccupationAppend(tRecX::readOperator("expectationValue"));
    std::string specDef = tRecX::readOperator("spectrum");

    if (iniDef == "hamiltonian")
        iniDef = hamDef;
    OperatorDefinition::setParameters(intDef); // need gauge radius for checking discretization setup
#ifdef _USE_UI_
    UiLayout::setPage();
#endif

    // time propagation
    string propMethod;
    double applyThreshold, accuracy, cutE, fixStep, tBeg = 0., tEnd = 0., tPrint, tStore;
#ifdef _USE_UI_
    UiLayout::setPage("Propagation");
#endif
#ifdef _USE_UI_
    UiLayout::setPage();
#endif
    std::string dum1, dum2;
    IndexConstraint::readAll(ReadInput::main, dum1, dum2);

    std::vector<double> surf;
    bool surfAscii(false);
    DiscretizationSurface::read(ReadInput::main,surf,surfAscii);
    int spectrumPoints=0;
    double minEnergy,maxEnergy;
    bool kGrid(false),computeSpectrum(ReadInput::main.found("Spectrum"));

    bool specOverwrite;
    std::string plotWhat, ampFiles;
    SpectrumPlot::read(ReadInput::main, plotWhat, specOverwrite, ampFiles);

    if (TsurffSource::recompute())
    {
        if(MPIwrapper::Size()>1)EXIT("-recompute option only on single processor");
        TsurffSource::moveSubdirs(ReadInput::main);
    }

    DiscretizationTsurffSpectra::readFlags(ReadInput::main,spectrumPoints,minEnergy,maxEnergy,kGrid,true);
    double energyScale=maxEnergy;
    if(spectrumPoints>0)energyScale=std::max(maxEnergy,4*math::pi/Units::convert(1.,"OptCyc"));

    int sampleForSpectrum=32;
    if(iSurff.isOn() and Units::convert(1,"OptCyc")<25.){
        PrintOutput::DEVwarning("HACK: introduced fine sampling for high-frequency iSurff, use -DEBUGsample to set sampling");
        sampleForSpectrum=128;
    }

    ReadInput::main.read("DEBUG","sample",sampleForSpectrum,tools::str(sampleForSpectrum),"sample for amplitude integration",1,"DEBUGsample");

    if(maxEnergy<=0.)tStore=-1.;
    else tStore=2.*math::pi/(energyScale*sampleForSpectrum);

    TimePropagator::read(ReadInput::main,tBeg,tEnd,tPrint,tStore,accuracy,cutE,fixStep,applyThreshold,propMethod);
    Parameters::setToTime(tBeg);

    std::string gauge=tRecX::readOperator("surfaceGauge");
    if(gauge=="automatic"){
        if((hamDef+intDef).find("LaserLength")!=std::string::npos)
            gauge="length";
        else if((hamDef+intDef).find("LaserF")!=std::string::npos)
            if(spectrumPoints>0 and tBeg<tEnd)
                ABORT("Hamiltonian "+hamDef+intDef+"contains laser field rather than vector potential, must define Operator: surfaceGauge");
    }
    if(gauge=="length")Algebra::addSpecialConstant("Rg",DBL_MAX);

    TsurffSource::allReads(ReadInput::main);

    std::string region=TsurffSource::nextRegion(ReadInput::main);
    if(region.find("unbound axes:")!=string::npos)
        PrintOutput::set(ReadInput::main.output()+"outspec");
    else if (region!="")
        ReadInput::main.outSubdir("S_"+region);
    STOP(all);

    int line(1);
    AxisTree(ReadInput::main,line);//HACK (but what for?)
    do {
        std::shared_ptr<OperatorTree> propOper,initialOper,hamOper;

        auto stopped=Timer::stopAll();
        if(stopped.size()>0)PrintOutput::DEVwarning(Sstr+"force-stopped timers"+stopped);
        START(all);
        START(setup);
        LOG_PUSH("execute");
        MPIwrapper::setCommunicator(Threads::all());

        timeCritical::suspend();
        // update spectral region
        region = TsurffSource::nextRegion(ReadInput::main);
        if (generateLinp())
            PrintOutput::set("cout");
        else
        {
            if (region.find("unbound") != string::npos)
            {
                // plot spectra from existing ampl files
                if (region.find("unbound axes:") != string::npos)
                {
                    // plotting does not work in parallel - use single thread
                    MPIwrapper::setCommunicator(Threads::single());
                    if (MPIwrapper::isMaster(Threads::all()))
                    {
                        PrintOutput::set(ReadInput::main.outputTopDir() + "outspec");

                        SpectrumPlot::allRegions(ReadInput::main.outputTopDir(), ReadInput::main);

                        PrintOutput::paragraph();
                        PrintOutput::message("spectra calculated from amplitude file(s), for recomputing amplitudes -recompute");
                    }
                }
                Timer::monitor(tools::str(tEnd)+"\tspectrum from ampl-file");
                goto Terminate;
            }
            else if (region != "" and
                     (spectrumPoints > 0 or
                      string(argv[1]).find(ReadInput::inputCopy) + ReadInput::inputCopy.length() == string(argv[1]).length()))
            {
                ReadInput::main.outSubdir("S_" + region);
                PrintOutput::set(ReadInput::main.output() + PrintOutput::outExtension);
            }
        }

        if (generateLinp())
            PrintOutput::off();

        std::string info="Master host = "+tools::cropString(platformSpecific::current_host()+" ");
        if(MPIwrapper::Size(Threads::all())>1)info+=" ("+tools::str(MPIwrapper::Size())+" processes) ";
        PrintOutput::subTitle(info);
        PrintOutput::paragraph();

        STARTDEBUG(input);
        LOG_PUSH("setup");
        LOG_PUSH("discretization");
        STARTDEBUG(setupDisc);

        std::shared_ptr<Discretization> mainD(Discretization::factory(ReadInput::main));
        PotentialTruncate::read(ReadInput::main,mainD->idx());
        BasisOrbital::addIndex("main", mainD->idx());
        if (ReadInput::main.flag("showMainIndex", "print main Index and stop"))
        {
            PrintOutput::message("AxisTree:\n" + mainD->axisTree()->Tree::str());
            PrintOutput::message("Index:\n" + mainD->idx()->str());
        }
        iSurff.checkCoordinates(mainD->idx()); // only few coordinate systems can do iSurff
        bool dipoleInitialContinuum=DipoleBoundContinuum::read(ReadInput::main,mainD->idx());
        StaticIonization::read(ReadInput::main);

        // with the index available, generat numerical orbitals
        STARTDEBUG(setupOrbital);
        BasisOrbitalNumerical::setup();
        STOPDEBUG(setupOrbital);

        STOPDEBUG(setupDisc);
        LOG_POP();
        LOG_PUSH("other");

        // ---------------------------------------------------------------------------------

        // Construct the discretization for the subregion, find all the sources feeding into this region and check if they exist
        std::vector<std::shared_ptr<TsurffSource>> sources;
        std::shared_ptr<Discretization> runD(mainD);

        // not nice: in case of surface flux, sf will carry the index, and must not be destroyed
        std::shared_ptr<SurfaceFlux> sf;
        if (region == "")
            mainD->print();
        else
        {
            // this is better now, but still very clumsy

            // calculation in subregion with a source
            std::string path = DiscretizationSurface::surfacePath(ReadInput::main.outputTopDir(), mainD->idx(), region);
            DiscretizationSurface *surface = new DiscretizationSurface(path);

            // surface index on file may become transformed in SurfaceFlux
            // use transformed for constructing this propagation index
            sf.reset(new SurfaceFlux(path, ReadInput::main));

            // spectral discretization (will be forked along radial spectral values)
            runD.reset(new DiscretizationTsurffSpectra(sf->idx(),ReadInput::main));

            sources.push_back(std::shared_ptr<TsurffSource>(new TsurffSource(dynamic_cast<DiscretizationTsurffSpectra*>(runD.get()),
                                                    ReadInput::main,surface,sf)));

            sources.back()->setTurnOff(ReadInput::main,tEnd);
            iSurff.addSource(sources.back());

            runD->print("", "DISCRETIZATION AND INCOMING FLUX");
            PrintOutput::newLine();
            MPIwrapper::Barrier();
            sources[0]->print(path);
            MPIwrapper::Barrier();
        }

        if (surf.size() > 0)
        {
            PrintOutput::paragraph();
            if (surf.size() == 1 and surf[0] > DBL_MAX / 2)
                PrintOutput::lineItem("Surfaces", "default (at complex scaling radius)");
            else
                PrintOutput::lineItem("Surfaces", Str("", "") + surf);
        }
        PrintOutput::paragraph();

        OperatorFloor3d::print();
        CoefficientsWriter::read(ReadInput::main);
        if (region != "")
            CoefficientsWriter::instance()->disable();
        // surface discretization for each unbound coordinate axis
        // NOTE: reverted to using same surface on all coordinates
        // NOTE: the logics below is messy and should be moved to a function of DiscretizationSurface
        vector<DiscretizationSurface *> discSurf;
        discSurf = DiscretizationSurface::all(ReadInput::main, runD, surf, region);

        vector<string> dipNames;
        string dipoleDef = Harmonics::dipoleDefinitions(ReadInput::main, mainD->idx()->coordinates(), hamDef+"+"+intDef, dipNames);
        Harmonics::read(ReadInput::main);
        PrintOutput::newRow();

        PrintOutput::title("OPERATORS");
        PrintOutput::paragraph();

        if (hamDef.find("[[eeInt6DHelium]]") != string::npos)
            PrintOutput::lineItem("Hamiltonian", Str(hamDef, "") + " (multipoleOrder=" + OperatorFloorEE::read(ReadInput::main) + ")");
        else
            PrintOutput::lineItem("Hamiltonian", hamDef);
        PrintOutput::newLine();

        if (mainD!=runD)
        {
            std::string subDef;
            // there is an ugly hack somewhere that <<EEhaCC>> must be last
            subDef = OperatorDefinition(hamDef, mainD->idx()->hierarchy()).tsurffDropTerms(runD->idx(), runD->idx());
            if (subDef != "")
                PrintOutput::lineItem("Region " + region, subDef);
            else
                PrintOutput::lineItem("Region " + region, "only spectral amplitudes");
            PrintOutput::newLine();
        }
        if (intDef != "")
        {
            PrintOutput::lineItem("Interaction", intDef);
            PrintOutput::newLine();
            if (region != "")
            {
                PrintOutput::lineItem("Region " + region, OperatorDefinition(intDef, mainD->idx()->hierarchy()).tsurffDropTerms(runD->idx(), runD->idx()));
                PrintOutput::newLine();
            }
        }

        if (initialKind == "manyBody")
            iniDef = hamDef;

        string iniName = iniDef;
        if (iniDef == hamDef)
            iniName = "Hamiltonian";
        else if (iniDef == "atBegin")
            iniName = "H(t=Begin)";
        PrintOutput::lineItem("HamInitial", iniName);
        PrintOutput::newLine();

        PrintOutput::newLine();
        if (paramDef != "")
        {
            PrintOutput::lineItem("Static field", paramDef);
            PrintOutput::newLine();
        }
        if (expecDef != "")
        {
            PrintOutput::lineItem("Expectation Values", expecDef);
            PrintOutput::newLine();
        }
        if (preconDef != "")
            PrintOutput::lineItem("Projection", preconDef);
        if (dipoleDef != "")
        {
            PrintOutput::newLine();
            PrintOutput::lineItem("Dipoles", dipoleDef);
            PrintOutput::newLine();
        }

        PotentialTruncate::print();
        double eShift=HamiltonianIon::readShift(mainD->idx());
        if(eShift!=0.){
            std::string reason=ReadInput::main.found("Shift")?" by input Shift:value":" such that lowest ionization threshold is =0";
            PrintOutput::subTitle("--- energy shifted by "+tools::str(eShift)+reason+" ---");
            if(hamDef.find("<<Overlap>>")!=std::string::npos){
                bool allow;
                ReadInput::main.read("Operator","allowOffset",allow,"false","allow <<Overlap>> in hamiltonian definition");
                if(not allow)ABORT("hamiltonian seems to contain offset by <<Overlap>> - is that intended? Shift:value is included automatically"
                          "\noverride by adding Operator:allowOffset=true to input");
            }

        }
#ifdef _DEVELOP_
        VectorValuedFunction::print();
#endif

        // ranges for operator parameters and initialize parameters
        ScanEigenvalue eigenScan(ReadInput::main);
        if (eigenScan.size() > 0)
            eigenScan.print();

        FloquetAnalysis floqAna(hamDef, ReadInput::main);
        EigenTrace eigenTrace(ReadInput::main);

        EigenSubspace eigSub(ReadInput::main);

        Pulse::current.output("", hamDef + intDef);

        Str strIni(initialKind, "");
        if (initialN != 0)
            strIni = strIni + " state[" + initialN + "] ";
        strIni = strIni + " for operator " + iniName;
        PrintOutput::lineItem("Initial state", strIni);
        PrintOutput::paragraph();

        string projConstraint;
        ReadInput::main.read("Project", "constraints", projConstraint, "", "constrain spectral projection");

        //        // set up possible spectral cuts
        //        SpectralCut specCut(runD.get(),ReadInput::main,hamDef);

        PrintOutput::title("OUTPUT");
        PrintOutput::lineItem("Directory", ReadInput::main.output());
        PrintOutput::paragraph();
        PrintOutput::paragraph();

        STARTDEBUG(setupPLot);
        std::shared_ptr<Plot> plot(new Plot(runD->idx(), ReadInput::main)); // set up plot, get the definitions from file
        plot->print();

        std::shared_ptr<PlotCoefficients> plotC(plot);
        STOPDEBUG(setupPLot);

        if (ReadInput::main.flag("showDiscretization", "print the discretization and stop"))
        {
            runD->print();
            exit(0);
        }

        // if no input specified, just print doc
        if (ReadInput::main.file() == "trecx.inp")
        {
            ReadInput::main.writeDoc();
            cout << "\nUsage: tRecX file[.inp] [-flags]" << endl;
            exit(0);
        }

        if (generateLinp())
        {
            ReadInput::main.writeDoc();
            cout << "\nList-of-inputs (linp) written to " + ReadInput::main.output() << endl;
            exit(0);
        }

        // create gradient operator, if specified (=0 else)
        OperatorAbstract *chanMap = 0, *grad = 0;
        string ChanAxis = OperatorMapChannelsSurface::axis(ReadInput::main);
        if (ChanAxis != "")
            grad = OperatorGradient::read(runD.get(), ReadInput::main);

        // constraints for initial state calculations (unmainained)
        DiscretizationConstrained::inputs(ReadInput::main);

        tRecX::read();
        tRecX::off("EigenSolver");
        tRecX::print();

        DensityOfStates dos(ReadInput::main);

        // further output
        HamiltonianIon::readShift(mainD->idx());
        bool saveWf;
        ReadInput::main.read("Output", "wavefunction", saveWf, "false", "save coefficients at print interval");
        if (ReadInput::main.flag("showIndex", "print Indices and stop"))
        {
            PrintOutput::outputLevel("full");
            PrintOutput::message("run Index\n" + runD->idx()->str());
            if (sf)
                PrintOutput::message("Surface Index\n" + sf->idx()->str());
            for (auto s : discSurf)
            {
                PrintOutput::message(s->name() + " Index\n" + s->idx()->str());
            }
            for (auto p : BasisOrbital::referenceIndex)
                PrintOutput::message(p.first + " Index\n" + p.second->str());
            EXIT("only showIndex");
        }

        // all inputs should be finished above these lines - please do not remove
        ReadInput::main.finish();
        PrintOutput::paragraph();
        STOPDEBUG(input);
        PrintOutput::outputLevel("full");
        PrintOutput::timerWrite("End of Input");
        //==== END OF INPUT ====================================================================

        if (floqAna.run(mainD->idx()))
            goto Terminate; // floquet analysis only
        if (eigenTrace.run(hamDef, mainD->idx(), ReadInput::main.output()))
            goto Terminate; // trace eigenvalues from guess

        // can use further cleanup... (get rid of Chandef/ChanAxis)
        std::string Chandef = "";
        if (ChanAxis != "")
            chanMap = new OperatorMapChannelsSurface(ReadInput::main, runD.get());

        LOG_POP();
        LOG_PUSH("Operators");

        START(buildOperators);
        vector<OperatorAbstract*>printOps;

        // there is an ugly hack somewhere that <<EEhaCC>> must be last
        tRecX::SetOperators(mainD->idx()->hierarchy(),hamDef,intDef,iniDef,"",dipNames,expecDef,runD.get(),propOper,hamOper,initialOper,printOps);

        PrintOutput::outputLevel("full"); // return to full output


        // Shift the full Hamiltonian spectrum by a specified channel energy
        OperatorMapChannelsSurface *mapch = static_cast<OperatorMapChannelsSurface *>(chanMap);
        if (mapch != nullptr && mapch->energyShift() != 0.)
        {
            std::string ovl;
            if (hamOper->idx()->isHybrid()) // hybrid basis needs special string
                ovl = "<1>(<<FullOverlap>>)+<1,0>[[haCC1]]+<0,1>[[haCC1]]";
            else
                ovl = "<<FullOverlap>>";
            ovl = tools::str(-mapch->energyShift()) + ovl;

            // HACK: As for now, it is not very clear what happens with OperatorTrees that got added/fused --> create two shift operators to be sure, don't dare to "delete" the pointers, as this may delete floors
            OperatorTree *shift = new OperatorTree("IonicShift", OperatorDefinition(ovl, hamOper->idx()->hierarchy()), hamOper->idx(), hamOper->idx());
            hamOper->add(shift);
            shift = new OperatorTree("IonicShift", OperatorDefinition(ovl, hamOper->idx()->hierarchy()), hamOper->idx(), hamOper->idx());
            propOper->add(shift);
            DEVABORT("Channel input is not longer maintained")
        }

        Harmonics::addDipolesToPlot(dipoleDef, *plot, printOps);
        STOP(buildOperators);
        LOG_POP();
        LOG_PUSH("other");

        STARTDEBUG(other);
        tRecX::PrintShow(printMatrices, showOperators, *propOper, *initialOper, printOps);

        // make sure inverse of overlap works correctly
        //DO NOT DISABLE - NEEDED FOR PROPAGATION (reason unclear at this point)
        if (propOper != 0)runD->idx()->testInverseOverlap();

        if (specDef != "")
        {
            PrintOutput::DEVwarning("Operator:spectrum temporarily out of service");
        }

        // output dos (if set up) and stop
        dos.output(propOper.get());

        STOPDEBUG(other);
        LOG_POP();
        LOG_POP();


        Parameters::restoreToTime();
        bool propagate=tBeg<tEnd;
        if (eigenSelect != "NONE")
        {
            if(OperatorHaCC::isHaCC(mainD->idx()))
                DEVABORT("at the moment, cannot compute full eigenvalues for HaCC");
            Timer::monitor("Eigensolver");
            Parameters::updateSpecial();
            std::shared_ptr<ProjectSubspace> proj=tRecX::projectPrecon(cutE,preconDef,printOps[1]->idx(),hamDef);
            tRecX::ComputeEigenvalues(eigenSelect, *plot, printOps, eigenVectors, eigenElements, proj);
            MPIwrapper::Barrier();
            Timer::monitor("Eigenvalues");
            goto Terminate;
        }

        else if (eigenScan.size() > 0)
        {
            // scan eigenvalues for a range of parameters
            Parameters::updateSpecial();
            DEVABORT("re-formulate for OperatorTree");
            //        Operator op(*propOper);
            //        eigenScan.setEigenSub(D,&op,dynamic_cast<const Operator*>(printOps[1]));
            //        eigenScan.scan();
        }

        else if (propagate or dipoleInitialContinuum)
        {
            STARTDEBUG(setProp);
            LOG_PUSH("DerivativeFlat");

            // no eigenvalue - time-propagate
            std::string precon(preconDef);
            if (preconDef=="default")precon=OperatorHaCC::isHaCC(mainD->idx())?"0.5<<Laplacian>>":hamDef;
            std::shared_ptr<DerivativeFlat> propDer(
                tRecX::SetDerivative(runD.get(), propOper, hamOper.get(),
                                     initialOper.get(), applyThreshold, cutE, precon, projConstraint, sources));


            LOG_POP();
            // set up the output
            TimePropagatorOutput out;
            tRecX::SetOutput(out,tPrint,tStore,surfAscii,discSurf,plotC.get(),grad,ChanAxis,Chandef,chanMap);
            out.setInterval(tBeg,tEnd);
            for(auto p: printOps)out.addExpec(p);
            if(saveWf)out.withApplyAndWrite(new OperatorIdentityTree(runD->idx()),"wf_t");
            if(dipoleDef!="")out.sampleExpec(1); /// when harmonics (dipoles) are desired, take ALL store points)

            PrintOutput::outputLevel("full"); // return to full output

            // checks and messages before start of time propagation
            std::shared_ptr<ChannelsSubregion> channels;
            if (propOper)
            {
                OperatorDefinition(propOper->def()).checkVariable("non-const parameters may have penalty in time-propagation, avoid!", propOper->name());

                LOG_PUSH("other");
                // set up channels
                if (runD != mainD and (runD->idx()->hierarchy() != "specX.Z.kX.Z"))
                {
                    //                    LOG_PUSH("ChannelsSubregion");
                    //                    runD->print();
                    //                    channels.reset(new ChannelsSubregion(runD.get(),hamOper.get(),region, ReadInput::main));
                    //                    LOG_POP();
                    //                    out.withChannelsSubregion(channels.get());
                    PrintOutput::DEVwarning("channels subregion disabled for development");
                }
                LOG_POP();
            }
            // time-propagator (for parallel code, create local derivative operator)
            LOG_PUSH("TimePropagator");

            std::unique_ptr<TimePropagator> prop;
            std::shared_ptr<DerivativeLocal>derLoc(new DerivativeLocal(propDer));
            OdeStep<LinSpaceMap<CoefficientsLocal>, CoefficientsLocal> *odeLoc;

            PrintOutput::outputLevel("full"); // return to full output

            if (propOper){
                //NOTE: if propDer->expI()!=nullptr OdeSplit will be built
                odeLoc = odeFactory<LinSpaceMap<CoefficientsLocal>, CoefficientsLocal>
                    (propMethod,derLoc.get(),accuracy,propDer->expI());
            }
            else
                odeLoc = odeFactory<LinSpaceMap<CoefficientsLocal>, CoefficientsLocal>("euler", derLoc.get(), accuracy);

            prop.reset(new TimePropagator(odeLoc, &out, accuracy, fixStep));

            LOG_POP();
            STOP(setup);

            // timer info for all setup
            PrintOutput::warningList();
            Parallel::timer();

            if (ReadInput::main.flag("DEBUGsetup", "stop after setup"))
            {
                STOPDEBUG(setProp);
                goto Terminate;
            }

            // initial wave function
            if (runD != mainD)
                initialKind = "ZERO";
            else if (initialKind=="" and iniDef != "atBegin" and initialKind != "manyBody"){
                initialKind = "Hinitial";
            }

            std::vector<std::shared_ptr<Coefficients>> Duals;
            if (printOps.size() > 1) {
                std::cout << "now at calc eigenvecs" << std::endl;
                std::vector<int> states = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
                std::cout << "printOps[0]=" << printOps[0]->name() << std::endl;
                const OperatorAbstract* specOp(printOps[1]);
                int kend = *std::max_element(states.begin(), states.end()) + 1;
                EigenSolver slv(-DBL_MAX, DBL_MAX, kend, true, true, false, "Lapack");
                slv.withSelect("SmallReal[10]");
                slv.fullVectors().compute(specOp, printOps[0]);
                slv.select("SmallReal[10]");
                slv.orthonormalize();
                std::vector<Coefficients*> rawDuals = slv.dualVectors();
                Coefficients* state3d = slv.rightVectors()[3];
                std::cout << state3d->str() << std::endl;
                Duals.reserve(rawDuals.size());
                for (const Coefficients* rawPtr : rawDuals) {
                    if (rawPtr) {
                        Duals.push_back(std::make_shared<Coefficients>(*rawPtr));
                    } else {
                        Duals.push_back(nullptr);
                    }
                }
            } else {
                std::cout << "calculating eigenvecs was not possible, set duals empty" << std::endl;
                Duals.clear();
            }

            // If there is a stored checkpoint, use it. Otherwise proceed normally
            Wavefunction wf;
            LOG_PUSH("InitialState");
            DerivativeFlatInhomogeneous::propdebug=true;
            STARTDEBUG(initial);
            PrintOutput::message("Initial state calculation");
            Timer::monitor("Initial["+initialKind+"]");
            wf = InitialState::get(runD.get(), initialKind, initialN, tBeg, *initialOper, propDer.get());
            if (dynamic_cast<DerivativeFlat*>(derLoc.get()))
                dynamic_cast<DerivativeFlat*>(derLoc.get())->project(*wf.coefs);
            if (propOper!=0 and expecDef.find("overInitial") != std::string::npos)
            {
                Coefficients c(wf.coefs->idx());
                wf.coefs->idx()->overlap()->apply(1., *wf.coefs, 0., c);
                std::shared_ptr<ProjectSubspace>proIni(new ProjectSubspace(std::vector<Coefficients *>(1, wf.coefs), std::vector<Coefficients *>(1, &c)));
                proIni->name() = "overInitial";
                out.addExpec(new ProjectSubspaceToDual(proIni));
            }
            LOG_POP();

            // propagation monitor settings
            string mess = runD->name() + " Tol=" + prop->str(1) + " " + Pulse::current.str(1);
            STOPDEBUG(initial);
            STOPDEBUG(setProp);
            LOG_PUSH("propagation");

            if (propOper==0)
            {
                // no actual propagation, only integration of source

                // add calculation of overlap
                out.sampleExpec(1); // put all into expec file
                out.addExpec(new OperatorTree("Overlap",OperatorDefinition::defaultOverlap(runD->idx()),
                                              runD->idx(),runD->idx()));
                out.checkGrowingNorm(false); // do not warn about norm>1

                //NOTE: we integrate by Euler, sample "sampl" times across shortest cycle in the spectrum
                prop->fixStep(2*math::pi/(sampleForSpectrum*energyScale));
            } else {

                for(auto s: sources){
                    if(s->str().find("turn off")!=std::string::npos)
                        PrintOutput::DEVwarning("source turn off for region "+region+"- undefined behavior");
                }
            }

            if(StaticIonization::read(ReadInput::main)){
                StaticIonization sfi(propDer.get(),ReadInput::main);
                sfi.computeRates();
                break;

            }
            else if(dipoleInitialContinuum){
                if(propOper!=0){
                    // replace main time-propagation with apply of operator
                    *wf.coefs=DipoleBoundContinuum::apply(*wf.coefs);
                }
                wf.time=tEnd;
                prop->propagate(&wf,wf.time,"Stop", Duals);                
            }
            else {
                // time-propagate
                PrintOutput::timerWrite("Start Propagation");
                if (wf.time < min(Pulse::gettEnd(), tEnd))
                    prop->propagate(&wf, min(Pulse::gettEnd(), tEnd), "Start", Duals);

                LOG_POP();
                LOG_PUSH("afterwards");
                START(timePropagate)
                if (Pulse::gettEnd() < tEnd)
                    prop->propagate(&wf, tEnd, "Stop", Duals);
                LOG_POP();
                STOP(timePropagate)
            }

            // amplitude calculation
            if(propOper==0 or region=="Rn1" or region=="Rn2"){
                MPIwrapper::Barrier();
                if(MPIwrapper::isMaster()){
                    if(sources.front()->channels())
                         sources.front()->channels()->unwind(*wf.coefs);

                    Coefficients res(*wf.coefs);
                    iSurff.addCorrection(res);
                    std::shared_ptr<Coefficients> joinedWf=Threads::join(res);
                    if(Threads::isMaster()){
                        ofstream ampl((ReadInput::main.output()+"ampl").c_str(),(ios_base::openmode) ios::beg|ios::binary);
                        joinedWf->write(ampl,true,"IndexFull");
                    }

                    PrintOutput::message("Photo-electron amplitudes on "+ReadInput::main.output()+"ampl");
                }
            }
            else if (region=="" and iSurff.isOn()){
                // save propOper and wave function into iSurff (if active)
                iSurff.endPropagation(wf.time,*wf.coefs,hamOper,propOper,propDer,discSurf.back()->sharedFromParent());
            }

            if (region == "Rn1" or region == "Rn2")
            {
                // single-ionization spectra
                std::shared_ptr<const Coefficients>joinedWf = Threads::join(*wf.coefs);
                if (Threads::isMaster())
                {
                    std::vector<std::string> ax = region == "Rn1" ? std::vector<std::string>(1, "kRn1") : std::vector<std::string>(1, "kRn2");
                    std::vector<std::string> use(1, "g");
                    if (region == "Rn1")
                        ax[0] = "kRn1";
                    Plot plt(joinedWf->idx(), ax, use);
                    plt.plot(*joinedWf, ReadInput::main.output() + "spec_single");
                    PrintOutput::message("single ionization spectrum on " + ReadInput::main.output() + "spec_single");
                }
            }

            PrintOutput::end();
            PrintOutput::paragraph();
            PrintOutput::subTitle("TimePropagator " + prop->info());
            MPIwrapper::setCommunicator(Threads::all());
            Timer::monitor("propagate region "+region);

        }
        else {
            PrintOutput::message(Str("tBeg =")+tBeg+">= tEnd="+tEnd+"--- no time propagation",0,true);
            Timer::monitor("no time propagation");
        }
        if(MPIwrapper::isMaster() and dipNames.size()>0)
            Harmonics(ReadInput::main,"expec"," ",false,1.);

        LOG_POP();

        STOP(all);

        //HACK clear static variables of the present run
        OperatorHaCC::clear();
        if(not iSurff.isOn())IndexOverlap::clear();

        PrintOutput::warningList();
        PrintOutput::timerWrite("End Propagation");
        Parallel::timer();
    } while ((computeSpectrum or ReadInput::main.found("Spectrum"))
             and (region != TsurffSource::nextRegion(ReadInput::main)
                  or TsurffSource::nextRegion(ReadInput::main).find("") != string::npos) and surf.size()>0 and spectrumPoints!=0);
    Parallel::clear();

Terminate:
    LOG_POP();
    LOG_POP();

    Timer::monitor("run completed",true); // force final message
    auto stopped=Timer::stopAll();

    if(stopped.size()>0){
        PrintOutput::DEVmessage(Sstr+"force-stopped timers, rewrite timer results: "+stopped);
        PrintOutput::timerWrite("Final");
        Parallel::timer();
    }


    PrintOutput::paragraph();
    PrintOutput::terminationMessages();

    RandomPotential::clearAll(); // remove old, all re-generate of random potential by the same name
    PrintOutput::title("done - "+ReadInput::main.output());
}
