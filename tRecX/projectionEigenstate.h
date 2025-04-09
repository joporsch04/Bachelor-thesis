        // std::vector<Coefficients*> rightVectorsVec;
        // for(auto k: states){  
        //     Coefficients* rightVectors_init(slv.rightVectors()[k]);
        //     rightVectors_init->setToConstant(1.);
        //     rightVectorsVec.push_back(rightVectors_init);
        // }
        // double norm = std::pow(rightVectors->idx()->overlap()->matrixElement(*rightVectors, *rightVectors).real(), 2)+std::pow(rightVectors->idx()->overlap()->matrixElement(*rightVectors, *rightVectors).imag(), 2);
        // std::cout << "Overlap norm of const vec " << k << ": " << norm << std::endl;
        // dualVectors->scale(1.0 / norm);
        // double norm_2 = std::pow(rightVectors->idx()->overlap()->matrixElement(*rightVectors, *rightVectors).real(), 2)+std::pow(rightVectors->idx()->overlap()->matrixElement(*rightVectors, *rightVectors).imag(), 2);
        // std::cout << "Overlap norm_2 of const vec " << k << ": " << norm_2 << std::endl;
        // std::cout << "norm after normalization: " << rightVectors->pNorm(2) << std::endl;
        for (auto* eigenVec : eigenVecs) {
            wfeig.coefs->idx()->overlap()->apply(1., *wfeig.coefs, 0., *eigenVec);
        }




                    // std::ifstream evecFile("/home/user/BachelorThesis/trecxtiptoe/tiptoe/0029/evec", std::ios::binary);
            // vector<Coefficients*> eigenVecs;
            // ifstream vecFile(ReadInput::main.output() + "evec", ios::binary);

            // std::ifstream vecFile("/home/user/BachelorThesis/trecxtiptoe/tiptoe_short/0062/evec", std::ios::binary);
        
            // std::vector<Coefficients*> eigen_Vecs_vector;
            // Coefficients* coef = new Coefficients(runD->idx()); // Use the appropriate Index object
            // if (!coef->read(vecFile, true)) { // Read from the binary file
            //     delete coef; // Clean up if reading fails
            //     break;
            // }
            // eigen_Vecs_vector.push_back(coef); // Add the successfully read Coefficients object to the vector
        



            // if (eigen_Vecs_vector.empty()) {
            //     std::cout << "No eigenstates found in eigenVecs." << std::endl;
            // }
            // std::complex<double> result = 0.0;
            // for (auto* eigen_Vecs_vector : eigen_Vecs_vector) {
            //     if (eigen_Vecs_vector == nullptr) {
            //         std::cout << "Encountered a null pointer in eigenVecs." << std::endl;
            //         continue;
            //     }
        
            //     std::cout << "Eigenstate: " << eigen_Vecs_vector->idx()->str() << std::endl;
            //     std::cout << "loop" << std::endl;
        
            //     // Compute the overlap
            //     auto overlap = eigen_Vecs_vector->innerProduct(wf.coefs);
            //     std::cout << "Overlap with eigenstates: " << overlap.real() << " + " << overlap.imag() << "i" << std::endl;
            // }

            // constant wavefunction for eigenstates
            // Wavefunction wfeig;
            // for (auto* eigenVec : eigenVecs) {
            //     wfeig.coefs->idx()->overlap()->apply(1., *wfeig.coefs, 0., *eigenVec);
            // }
            // std::shared_ptr<ProjectSubspace> proIni(new ProjectSubspace(std::vector<Coefficients*>(1, wfeig.coefs), eigenVecs));
            // out.addExpec(new ProjectSubspaceToDual(proIni));


            
            //eigenVecs = tRecX::ComputeEigenvalues_new(eigenSelect, printOps, eigenVectors, eigenElements, proj, eigenVectorsAscii);


            , eigenVectorsAscii;
    std::vector<Coefficients*> eigenVecs;
    eigenVectorsAscii = true;





















    void ProjectSubspace::_construct_new(std::vector<const Coefficients *> Vectors, std::vector<const Coefficients *> Duals,size_t BeginOrthonormal)
{
    if(not Vectors.size())DEVABORT("must not construct ProjectSubspace w/o any Vectors");
    if(Vectors.size()!=Duals.size())DEVABORT(Sstr+"unequal number of Vectors and Duals"+Vectors.size()+Duals.size());
    _name__="Project"+tools::str(Vectors.size());
    if(Vectors.size()==0)return;
    if(BeginOrthonormal*Vectors.size()>1.e6)PrintOutput::DEVmessage(Sstr+"setting up large projector from"+BeginOrthonormal+"non-ON vectors");
    _iIndex=Vectors[0]->idx();
    _jIndex=_iIndex;

    // check duals
    if(Vectors[0]->idx()!=Duals[0]->idx())DEVABORT(Sstr+"unequal Index on Vectors and Duals"+Vectors[0]->idx()+Duals[0]->idx());
    checkDuals({Vectors.begin()+BeginOrthonormal,Vectors.end()},{Duals.begin()+BeginOrthonormal,Duals.end()},1.e-8);

    std::vector<Coefficients*> copiedDuals;
    for (const auto* c : Duals)copiedDuals.push_back(const_cast<Coefficients*>(c));
    for (auto* c : copiedDuals){
        c->setToConstant(1/990.); 
        std::cout << "copiedDuals: " << c->pNorm(2) << std::endl;   
    }
    std::vector<const Coefficients*> copiedDualsPtrs;
    for (const auto* c : copiedDuals) {
        copiedDualsPtrs.push_back(c);
    }
    std::vector<std::vector<std::vector<const Coefficients*> > > gVecs=groupByNonzeros(Vectors,copiedDualsPtrs,_sorting);

    // build a grouped subspace index
    Index* idx=new Index(std::vector<const BasisAbstract*>(1,BasisAbstract::factory("Vector:"+tools::str(gVecs[0].size()))),{"Subspace"});
    for(size_t k=0;k<idx->basis()->size();k++)
        idx->childReplace(k,new Index(std::vector<const BasisAbstract*>(1,BasisAbstract::factory("Vector:"+tools::str(gVecs[0][k].size()))),{"SubSubspace"}));
    std::vector<std::string> dum;


    // for now, we consider projections as global (can be improved)
    _subspaceIndex=idx;
    _subspaceIndex->setFloorAuto(dum);
    _subspaceIndex->sizeCompute();
    ParallelLayout::setOwnersRoot(_subspaceIndex->root(),MPIwrapper::master());

    _subspaceC.reset(new Coefficients(_subspaceIndex));
    
    _mapTo.reset  (new OperatorTree("mapToFullFromContracted",_iIndex,_subspaceIndex));
    _mapFrom.reset(new OperatorTree("mapFromFullToContracted",_subspaceIndex,_iIndex));

    for(size_t k=0;k<gVecs[0].size();k++){
        _mapTo->childAdd(new OperatorMapCoefficients("block",_iIndex,_subspaceIndex->child(k),gVecs[0][k]));
        _mapFrom->childAdd(new OperatorMapCoefficients("block",_subspaceIndex->child(k),_iIndex,gVecs[1][k]));
    }
    _mapTo->purge(1.e-12);
    _mapFrom->purge(1.e-12);

    if(BeginOrthonormal>0){
        if(_opC)DEVABORT("cannot use _lu and transformation");
        std::vector<Eigen::Triplet<std::complex<double>>> list;
        for(size_t j=0;j<BeginOrthonormal;j++){
            for(size_t i=0;i<Duals.size();i++){
                list.push_back(Eigen::Triplet<std::complex<double> >(i,j,Duals[i]->innerProduct(Vectors[j],true)));
                if(i>=BeginOrthonormal)
                    list.push_back(Eigen::Triplet<std::complex<double> >(j,i,Duals[j]->innerProduct(Vectors[i],true)));
            }
        }
        // fill up diagonal
        for(size_t k=BeginOrthonormal;k<Vectors.size();k++)
            list.push_back(Eigen::Triplet<std::complex<double> >(k,k,1));

        Eigen::SparseMatrix<std::complex<double>> sov(Duals.size(),Vectors.size());
        sov.setFromTriplets(list.begin(),list.end());

        // permute the sparse matrix into _sorting
        Eigen::PermutationMatrix<Eigen::Dynamic> perm(sov.rows());
        for(size_t k=0;k<_sorting.size();k++)perm.indices()[_sorting[k]]=k;
        sov=sov.twistedBy(perm);

        // get the sparse LU decomposition
        _lu.reset(new Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>,Eigen::COLAMDOrdering<int>>(sov));
        _luTranspose.reset(new Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>,Eigen::COLAMDOrdering<int>>(sov.transpose()));

    }


    ParallelLayout::determineAllHosts(_mapTo.get());
    ParallelLayout::determineAllHosts(_mapFrom.get());
    // make sure norms and loads are sync'd
    ParallelOperator::sync(_mapFrom.get());
    ParallelOperator::sync(_mapTo.get());

    if(not verify())DEVABORT("basic projector does not work");
}
