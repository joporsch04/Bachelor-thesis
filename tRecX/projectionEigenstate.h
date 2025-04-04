#ifndef PROJECTIONONEIGENSTATE_H // prevents multiple inclusions
#define PROJECTIONONEIGENSTATE_H 

#include "operatorTree.h"
#include "eigenSolver.h"
#include "timePropagatorOutput.h"
#include "projectionSingle.h"
#include "projectionEigenstate.h"

class ProjectionOnEigenstate : public ProjectionOperator {
public:
    ProjectionOnEigenstate(const OperatorAbstract* hamiltonian, const Index* idx, unsigned int numEigenstates, double energyCutoff)
        : ProjectionOperator({}, idx), _hamiltonian(hamiltonian), _numEigenstates(numEigenstates), _energyCutoff(energyCutoff) {
        computeEigenstates();
    }

    void computeEigenstates() {
        EigenSolver solver(-_energyCutoff, _energyCutoff, _numEigenstates);
        solver.compute(_hamiltonian);

        _eigenvectors = solver.rightVectors();
    }


    // i need a method that "makes" the projection operators with the given eigenstates and returns it

    std::vector<std::shared_ptr<ProjectionSingle>> createProjectionOperators() const {
        std::vector<std::shared_ptr<ProjectionSingle>> projectionOperators;

        for (size_t i = 0; i < _eigenvectors.size(); ++i) {
            auto projectionSingle = std::make_shared<ProjectionSingle>(_eigenvectors[i], idx(), idx());
            projectionOperators.push_back(projectionSingle);
        }

        return projectionOperators;
    }



private:
    const OperatorAbstract* _hamiltonian;
    unsigned int _numEigenstates;
    double _energyCutoff;
};

#endif // PROJECTIONONEIGENSTATE_H



// next: add operator to this function:
//
// void TimePropagatorOutput::addExpec(OperatorAbstract* Op){
//     _expecOp.push_back(Op);
//     openExpec(ReadInput::main.output()+"expec");
// }
//
// i think it is called in run_tRecX.cpp right here:
//
// if (propOper==0)
// {
//     // no actual propagation, only integration of source

//     // add calculation of overlap
//     out.sampleExpec(1); // put all into expec file
//     out.addExpec(new OperatorTree("Overlap",OperatorDefinition::defaultOverlap(runD->idx()),
//                                   runD->idx(),runD->idx()));
//     out.checkGrowingNorm(false); // do not warn about norm>1

//     //NOTE: we integrate by Euler, sample "sampl" times across shortest cycle in the spectrum
//     prop->fixStep(2*math::pi/(sampleForSpectrum*energyScale));
// } else {