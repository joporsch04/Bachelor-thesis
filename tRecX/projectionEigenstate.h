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