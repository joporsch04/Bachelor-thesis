// tRecX = tSurff+irECS - a universal Schroedinger solver

// tRecX = tSurff+irECS - a universal Schroedinger solver
// Copyright (c) 2015 - 2024 by Armin Scrinzi (armin.scrinzi@lmu.de)
// 
// This program is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free Software Foundation; 
// either version 2 of the License, or (at your option) any later version.
// End of license

#include "timePropagatorOutput.h"

#include <complex>
#include "coefficientsSparse.h"
#include "operatorHaCC.h"
#include "threads.h"
#include "timer.h"
#include "timeCritical.h"
#include "checkpoint.h"
#include "index.h"
#include "printOutput.h"
#include "plot.h"
#include "readInput.h"
#include "channelsSubregion.h"

#include "operatorDefinition.h"

#include "parallelOperator.h"
#include "timePropagatorOutput.h"
#include "units.h"

#ifdef _USE_FFTW_
#include "constants.h"
#include <fftw3.h>
#endif
using namespace std;

TIMER(outWrite,)
TIMER(outWrite1,)
TIMER(outWrite2,)
TIMER(outWrite3,)
TIMER(outWrite4,)
TIMER(outWrite5,)

static double timPrev=-DBL_MAX;
static double deltaSum=0.;
static int ntime=0;

static std::map<std::string,std::complex<double>>compExp;

int TimePropagatorOutput::obj_count = 0;

bool checkForGrowingNorm=false;

void TimePropagatorOutput::openExpec(std::string ExpecFile){
    if(expecStream)return; // is already open
    if(not MPIwrapper::isMaster())return; // only master writes expec
    Checkpoint chPt(ExpecFile.substr(0,ExpecFile.rfind("/")));

    if(ExpecFile!=""){
        // only master writes expectation values - do not create empty files
        if(chPt())std::rename((ExpecFile).c_str(),(ExpecFile+"_crashed").c_str());
        expecStream=new ofstream();
        expecStream->open(ExpecFile.c_str(),(ios_base::openmode) ios::beg);
        ifstream ifil((ExpecFile+"_crashed").c_str(),(ios_base::openmode) ios::beg);
        if(ifil){
            std::string line;
            while(getline(ifil,line)){
                if(line[0]!='#'){
                    std::stringstream slin(line);
                    double time;
                    slin>>time;
                    if(time+_printInterval*1.e-10>chPt.time())break;
                }
                *expecStream<<line<<std::endl;
            }
        }
        std::remove((ExpecFile+"_crashed").c_str());
    }
}


TIMER(propIntern,)
TimePropagatorOutput::TimePropagatorOutput(
    double PrintInterval,std::string WriteDir, const PlotCoefficients *WfPlot, std::vector<const OperatorAbstract *> Write,
    double WriteInterval, bool WriteAscii)
    :expecStream(0),_expecSample(0),countPrint(0),countWrite(0),dir(WriteDir),wfPlot(WfPlot),_channels(0),_checkGrowingNorm(true),
    _tStart(DBL_MAX),_tEnd(-DBL_MAX),_writeInterval(WriteInterval),_printInterval(PrintInterval),_writeOp(Write),_discSpec(0)
{

    obj_count++;
    _timer=new Timer("propIntern"+tools::str(obj_count),"","timePropagator");
#ifndef _DEVELOP_
    _timer->setShow(0);
#endif
    if(wfPlot!=0){
        double t=_printInterval;
        if(Units::isDefined("OptCyc"))t=Units::convert(t,"DEFAULT_SYSTEM","OptCyc");
        wfPlot->setPlotInterval(t,false);
    }
    

    // storage for converting functions before write
    for(unsigned int k=0;k<_writeOp.size();k++){
        temp.push_back(Coefficients(_writeOp[k]->idx()));
        temp.back().treeOrderStorage();
    }

    // always write first function
    _lastTimeWritten=-DBL_MAX;

    // minimal time-interval for writing
    if(WriteInterval<0.)_writeInterval=_printInterval;

}
void TimePropagatorOutput::setInterval(double TBeg, double TEnd){
    _tStart=TBeg;
    _tEnd=TEnd;
    if(dir!="" and MPIwrapper::isMaster()){
        // if there is a checkpoint before end of time-propagation, copy to new output files
        Checkpoint chPt(dir);
        if(chPt.time()<_tEnd){
            for(unsigned int k=0;k<_writeOp.size();k++) {
                const Index* jI=_writeOp[k]->idx();
                Coefficients cK(jI);
                auto jK=Threads::join(cK);
                if (Threads::isMaster()){
                    std::string fileName = _writeOp[k]->name();
                    if(chPt())std::rename((dir+fileName).c_str(),(dir+fileName+"_crashed").c_str());

                    streamBin.push_back(new ofstream((dir+fileName).c_str(),(ios_base::openmode) ios::beg|ios::binary));
                    jK->idx()->write(*streamBin.back()); // coefficients outputs go with Index info

                    if(chPt()){
                        // copy crashed into new up to checkpoint time
                        ifstream ifil((dir+fileName+"_crashed").c_str(),(ios_base::openmode) ios::beg|ios::binary);
                        if(ifil){
                            Coefficients fC(jK->idx());
                            bool head=true;
                            double timeOnFile;
                            do{
                                if (not fC.read(ifil,head))break;
                                tools::read(ifil,timeOnFile);
                                if(timeOnFile>=chPt.time())break;
                                fC.write(*streamBin.back(),false);
                                tools::write(*streamBin.back(),timeOnFile);
                                head=false;
                            } while(true);
                            std::remove((dir+fileName+"_crashed").c_str());
                        }
                        else {
                            streamBin.pop_back();
                            PrintOutput::warning(Sstr+"has checkpoint at"+chPt.time()+"but no matching output file"+(dir+fileName)+" - file will not be written");
                        }
                    }
                }
            }
        }
    }
}


TimePropagatorOutput & TimePropagatorOutput::withApplyAndWrite(const OperatorAbstract *Op, std::string FileName){
    _writeOp.push_back(Op);
    temp.push_back(Coefficients(_writeOp.back()->idx()));

    if(MPIwrapper::isMaster()){
        if(FileName=="")FileName=Op->name();
        Coefficients cK(Op->idx());
        auto jK=Threads::join(cK);
        if (Threads::isMaster()){
            streamBin.push_back(new ofstream((dir+FileName).c_str(),(ios_base::openmode) ios::beg|ios::binary));
            if(not streamBin.back()->is_open())ABORT("could not open output file "+dir+FileName);
            jK->idx()->write(*streamBin.back());
        }
    }
    return *this;
}

bool inClosedInterval(double X,double From,double To,double Eps=1.e-10){
    if(Eps==0)return From<=X and X<=To;
    return From-Eps*max(abs(From),1.)<X and X<To+Eps*max(abs(To),1.);
}
void TimePropagatorOutput::printStart(double accuracy, double FixStep, double CutEnergy, std::string ODEname)
{
    if(_tStart==0. and _tEnd==0.)return;

    PrintOutput::title("PROPAGATION PARAMETERS");
    PrintOutput::paragraph();

    PrintOutput::newRow();
    PrintOutput::rowItem("begin");
    PrintOutput::rowItem("end");
    PrintOutput::rowItem("print");
    PrintOutput::rowItem("store");
    PrintOutput::rowItem("fixStep");
    PrintOutput::rowItem("accuracy");
    PrintOutput::rowItem("cutEnergy");
    PrintOutput::rowItem("method");

    PrintOutput::newRow();
    PrintOutput::rowItem(_tStart);
    PrintOutput::rowItem(_tEnd);
    PrintOutput::rowItem(_printInterval);
    if(_expecSample>1)PrintOutput::rowItem(tools::str(_writeInterval,2)+"["+tools::str(_expecSample)+"]");
    else if(_writeInterval==0)PrintOutput::rowItem("all");
    else                      PrintOutput::rowItem(_writeInterval);

    if(FixStep>0.){
        PrintOutput::rowItem(FixStep);
        PrintOutput::rowItem("no control");
    } else {
        PrintOutput::rowItem("variable");
        PrintOutput::rowItem(accuracy);
    }
    PrintOutput::rowItem(CutEnergy);
    PrintOutput::rowItem(ODEname);

    PrintOutput::paragraph();

    if(_expecSample>1 and _writeInterval>0)
        PrintOutput::message(Sstr+"Expectation values sampled only every"+_expecSample+"'th store time");
    if(_expecSample==0 and _writeInterval>0)
        PrintOutput::message(Sstr+"Expectation values will only be computed at TimePropagation:print times");

}

static std::complex<double> expectationValue(int IOp, std::vector<OperatorAbstract*> Ops, double Time, const Coefficients* Wf,bool Normalize){
    Ops[IOp]->update(Time,Wf);
    complex<double> expec=Ops[IOp]->matrixElementUnscaled(*Wf,*Wf);
    expec=Threads::sum(expec);
    // compute norm of overlap in list
    if(Normalize){
        std::complex<double> nrm=1.;
        if(Ops[IOp]->name().find("Ovr(")==std::string::npos){
            for(auto o: Ops){
                if(o->name().find("Ovr(")!=std::string::npos){
                    nrm=Threads::sum(o->matrixElementUnscaled(*Wf,*Wf));
                    break;
                }
            }
        }
        // initially amplitude is exactly = 0
        if(nrm!=0.)expec/=nrm;
    }
    return expec;
}

static std::complex<double> eigenProjection(int IOp, std::vector<OperatorAbstract*> Ops, double Time, const Coefficients* Wf,bool Normalize, std::vector<std::shared_ptr<Coefficients>> Duals){
    Ops[IOp]->update(Time,Wf);
    if (IOp > 1 && !Duals.empty() && IOp - 2 < Duals.size()) {
        const Coefficients* constdual_raw = Duals[IOp - 2].get();
        complex<double> expec=Ops[IOp]->matrixElementUnscaled(*constdual_raw,*Wf);
        expec=Threads::sum(expec);
        if(Normalize){
            std::complex<double> nrm=1.;
            if(Ops[IOp]->name().find("Ovr(")==std::string::npos){
                for(auto o: Ops){
                    if(o->name().find("Ovr(")!=std::string::npos){
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
            if(Ops[IOp]->name().find("Ovr(")==std::string::npos){
                for(auto o: Ops){
                    if(o->name().find("Ovr(")!=std::string::npos){
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

static bool equalCoefficients(const Coefficients* A, const Coefficients* B){
    if(A==B)return true;
    if(A==0 or B==0)return false;
    if(not A->idx()->treeEquivalent(B->idx()))return false;
    if(A->values()!=B->values())return false;
    return true;
}


void TimePropagatorOutput::print(const Coefficients *Wf, double Time, double TimeCPU, std::vector<std::shared_ptr<Coefficients>> Duals){

    timeCritical::suspend();
    double growingNorm=1.;
    double time=Time;
    if(_printInterval!=DBL_MAX){

        if(countPrint==0){
            PrintOutput::title("TIME PROPAGATION ("+ReadInput::main.output()+")");
            

            if(Time>_tStart){
                PrintOutput::paragraph();
                PrintOutput::message(Sstr+"resuming from time"+Time+"after start at"+_tStart);
            }

            if(_expecOp.size()>2 or (_expecOp.size()==2 and _expecOp[1]->name()!="H0")){
                // list definitions of expectation value operators
                PrintOutput::paragraph();
                for(size_t k=_expecOp.size()==2?1:2;k<_expecOp.size();k++){
                    PrintOutput::lineItem("<"+_expecOp[k]->name()+">",_expecOp[k]->def()+" at hierarchy "+_expecOp[k]->idx()->hierarchy());
                    PrintOutput::newLine();
                }
            }

            if(_expecOp.size()>1){
                PrintOutput::paragraph();
                auto tmp=_expecOp[1]->matrixElementUnscaled(*Wf,*Wf).real();
                PrintOutput::lineItem("Initial <"+_expecOp[1]->name()+">",tmp,"",14);
                PrintOutput::newLine();
            }

            PrintOutput::paragraph();
            PrintOutput::newRow();
            PrintOutput::rowItem("   CPU ");
            PrintOutput::rowItem(" (%)");
            PrintOutput::rowItem("   Time");
            for(unsigned int i=0;i<_expecOp.size();i++){
                if(_expecOp[i]->name().length()>16)
                    PrintOutput::rowItem("<"+_expecOp[i]->name().substr(0,13)+"...>");
                else
                    PrintOutput::rowItem("<"+_expecOp[i]->name()+">");
            }
            //HACK re-establish distribution pattern of operators
            for(size_t k=0;k<_expecOp.size();k++){
                ParallelOperator::setDistribution(_expecOp[k]);
            }
            for(auto o: _expecOp){
                o->update(Time,Wf);
                if(o->def().find("MeanEE")!=string::npos)dynamic_cast<OperatorTree*>(o)->updateNonLin(Time,Wf);
            }
        }
        PrintOutput::newRow();
        PrintOutput::rowItem(TimeCPU);
        PrintOutput::rowItem((Time-_tStart)/(_tEnd-_tStart)*100.);

        // write near-zero time as 0
        if(abs(time)<abs(_tStart-_tEnd)*1.e-10)time=0.;
        PrintOutput::rowItem(time,6);
        

        double complexEps=1.e-6;
#ifdef _DEVELOP_
        complexEps=1.e-9; // more sensitive for developer
#endif
        for(unsigned int i=0;i<_expecOp.size();i++){
            std::complex<double>expec=expectationValue(i,_expecOp,Time,Wf,false);

            PrintOutput::rowItem(real(expec),14);
            if( abs(expec)>complexEps
                and abs(expec.imag())>abs(std::abs(expec))*complexEps
                and abs(expec.imag())>complexEps){
                if(_expecOp[i]->name().find("Ovr")==std::string::npos and
                    std::abs(compExp[_expecOp[i]->name()].imag())<std::abs(expec.imag())){
                    compExp[_expecOp[i]->name()]=expec;
                }
            }
        }
        for(auto c: compExp){
            PrintOutput::DEVwarning("Complex expectation value of operator "
                                        +c.first+" (threshold="+tools::str(complexEps)+")",5,0,
                                    "complex scaled operators coincide with the unscaled up to the scaling radius R0"
                                    "\nat r=R0 the operator can violate hermiticity"
                                    "\nif the wave function has significant amplitude there, complex expectation can appear"
                                    "\nlarge imaginary parts may indicate flawed definition of the operator");
        }

        double t=time;
        if(Units::isDefined("OptCyc"))t=Units::convert(time,"DEFAULT_SYSTEM","OptCyc");
        vector<string> head(1,string("time="+tools::str(time,8)));
        auto joinedWf=Threads::join(*const_cast<Coefficients*>(Wf));
        // only plot if either standard parallel Wf or joinedWf on master of Threads
        if(wfPlot and  (equalCoefficients(joinedWf.get(),Wf) or Threads::isMaster()))
            wfPlot->plot(*joinedWf,dir+wfPlot->briefName()+tools::str(t,5),head);

        PrintOutput::flush();
        if(growingNorm-1.>1.e-7)checkForGrowingNorm=true;
        if(Threads::isMaster())Checkpoint::write(dir,Time,joinedWf.get());
        write(Wf,Time,true, Duals); // force write whenever there is print
        // write_projec(Wf,Time,true, Duals); // force write whenever there is print // why do i need an extra write part here, it is already being written in propagate bzw propagate_intern
        flush();

        if(std::abs(time-_tEnd)<std::abs(_tEnd-_tStart)*1e-10){
            if(compExp.size()){
                std::vector<std::string> info(1,"Imaginary expectation values");
                info.push_back("Operator: value where maximal imaginary part appears");
                for(auto c: compExp){
                    info.push_back(c.first+": "+tools::str(c.second));
                }
                info.push_back("");
                info.push_back("Complex expectation values can appear with complex scaling.");
                info.push_back("Physical interpretation of expecation values is limited");
                info.push_back("to within the magnitude of the imaginary part");

                PrintOutput::info(info);
                compExp.clear();
            }
        }

        countPrint++;
        timeCritical::resume();
    }
}

void TimePropagatorOutput::flush() const {
    for(auto b: streamBin)b->flush();
    for(auto a: streamAsc)a->flush();
    if(expecStream)expecStream->flush();
    PrintOutput::flush();
}

void TimePropagatorOutput::close(){
    if(expecStream)expecStream->flush();
    for(unsigned int c=0;c<streamBin.size();c++){
        streamBin[c]->close();
        if(streamBin[c]) delete streamBin[c];
        streamBin[c] = 0;
    }
    for(unsigned int c=0;c<streamAsc.size();c++){
        streamAsc[c]->close();
        if(streamAsc[c]) delete streamAsc[c];
        streamAsc[c] = 0;
    }
}

void TimePropagatorOutput::write(const Coefficients *C, double Time, bool Force, std::vector<std::shared_ptr<Coefficients>> Duals){  
    if(_lastTimeWritten>=Time)return; // never write backward or duplicate times
    if(_channels)_channels->average(C, Time);
    if(not Force and Time<nextWriteTime())return;
    STARTDEBUG(outWrite);
    _lastTimeWritten=Time;

    // header on expec file
    if(expecStream and not expecStream->tellp()){
        for(unsigned int k=0;k<_expecOp.size();k++)*expecStream<<"# "+_expecOp[k]->name()<<endl;
        *expecStream<<"#      Time            CPU ";
        for(unsigned int i=0;i<_expecOp.size();i++)*expecStream<<setw(20)<<"<"+_expecOp[i]->name()+">";
        for(unsigned int i=2;i<_expecOp.size();i++)*expecStream<<setw(20)<<"Re{<H0:"<<i-2<<"|psi>}";
        for(unsigned int i=2;i<_expecOp.size();i++)*expecStream<<setw(20)<<"Imag{<H0:"<<i-2<<"|psi>}";
        *expecStream<<endl;
    }

    double t=Units::isDefined("OptCyc")? Units::convert(Time,"DEFAULT_SYSTEM","OptCyc") : Time;
    if(wfPlot){
        // on this level, true plots will only be done if in append mode
        const Plot* plot=dynamic_cast<const Plot*>(wfPlot);
        if(not plot or plot->isAppend()){
            std::shared_ptr<const Coefficients> joinedC=Threads::join(*const_cast<Coefficients*>(C));
            // only Threads master writes
            if(Threads::isMaster())
                wfPlot->plot(*joinedC,dir+wfPlot->briefName(),std::vector<std::string>(),tools::str(t,12));
        }
    }

    // apply operators and write wave functions
    for(unsigned int k=0;k<_writeOp.size();k++){
        if(timPrev!=-DBL_MAX){
            deltaSum+=Time-timPrev;
            ntime++;
        }
        timPrev=Time;
        const_cast<OperatorAbstract*>(_writeOp[k])->axpy(1.,*C,0.,temp[k],Time);
        std::shared_ptr<Coefficients> joinedTemp=Threads::join(temp[k]);
        if(Threads::isMaster()){
            if(_writeOp[k]->name().find("surface_")==0 and OperatorHaCC::isHaCC(_writeOp[k]->jdx())){
                joinedTemp->scale(1./MPIwrapper::Size());
            }
            joinedTemp->write(*streamBin[k],false);
            tools::write(*streamBin[k],Time);
            if(streamAsc.size()>k){
                if(_writeOp[k]->name()=="surface_Rn"){
                    joinedTemp->purgeNearZeros(1.e-10);
                }
                *streamAsc[k]<<Time;
                joinedTemp->print(*streamAsc[k]);
                streamAsc[k]->flush();
            }
        }
    }

    // expectation values (if specified)
    if((Force or (_expecSample and not (countWrite%_expecSample))) and _expecOp.size()){
        if(expecStream){
            *expecStream<<setw(20)<<setprecision(12)<<Time; // time column
            *expecStream<<setw(12)<<setprecision(4)<<_timer->secs(); // CPU column
            for(unsigned int i=0;i<_expecOp.size();i++){
                std::complex<double>expec=expectationValue(i,_expecOp,Time,C,false);

                MPIwrapper::Barrier();
                double expecRe=Threads::sum(expec.real());
                *expecStream<<setw(20)<<setprecision(12)<<expecRe;
            }
            for(unsigned int i=2;i<_expecOp.size();i++){
                std::complex<double>eigenProjec=eigenProjection(i,_expecOp,Time,C,false, Duals);

                MPIwrapper::Barrier();
                double eigenProjecRe=Threads::sum(eigenProjec.real());
                *expecStream<<setw(20)<<setprecision(12)<<eigenProjecRe;
            }
            for(unsigned int i=2;i<_expecOp.size();i++){
                std::complex<double>eigenProjec=eigenProjection(i,_expecOp,Time,C,false, Duals);

                MPIwrapper::Barrier();
                double eigenProjecImag=Threads::sum(eigenProjec.imag());
                *expecStream<<setw(20)<<setprecision(12)<<eigenProjecImag;
            }
            *expecStream<<std::endl;
        }
    }
    if(_checkGrowingNorm and C->idx()->overlap()){
        double growingNorm=C->idx()->overlap()->matrixElementUnscaled(*C,*C).real();
        growingNorm=Threads::sum(growingNorm);
        if(growingNorm>1.2){
            PrintOutput::warning(Str("norm>1, possible instability - try smaller scaling angle or smaller operator threshold"),1,0,
                                 "Possible reasons - trouble  shoot: \
                                 \n - scaling angle too large - use smaller\
                                 \n - TimePropagator:accurcy or fixStep too large \
                                 \n - basis in absorption range too small - use larger \
                                 \n - polynomial order too high - use lower \
                                 \n - spectral cut in time-propagator fails - increase cut energy \
                                                  "
                                 );
        }
        if(growingNorm>10. and not ReadInput::main.flag("DEBUGignoreNorm","continue calculation even if norm grows strongly"))
            ABORT("fatal growth of norm, ignore by -DEBUGignoreNorm");
        if(growingNorm<=1.)checkForGrowingNorm=false;
    }
    countWrite++;
    STOPDEBUG(outWrite);
}


void TimePropagatorOutput::addExpec(OperatorAbstract* Op){
    _expecOp.push_back(Op);
    openExpec(ReadInput::main.output()+"expec");
}

void TimePropagatorOutput::coefsFFT() {
    if(not OperatorAbstract::flat)return;
    if(coefsT.size()==0)return;
#ifdef _USE_FFTW_
    ofstream fftStream((ReadInput::main.output()+"coefs").c_str(),(ios_base::openmode) ios::beg);
    PrintOutput::paragraph();
    unsigned int nOmega=coefsT[0].size()/400;
    PrintOutput::message("Fourier transform of Coefficients on file "+ReadInput::main.output()+"coefs");

    fftw_complex *in, *out;
    fftw_plan p=0;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * coefsT[0].size());
    out =(fftw_complex*) fftw_malloc(sizeof(fftw_complex) * coefsT[0].size());
    vector<double> smooth(coefsT[0].size());
    for  (unsigned int k=0;k<smooth.size();k++)
        smooth[k]=pow(sin(k*math::pi/double(smooth.size())),8);
    for(unsigned int k=0;k<coefsT.size();k++){
        p = fftw_plan_dft_1d(coefsT[0].size(), in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        for (unsigned int l=0;l<coefsT[k].size();l++){*(in+l)[0]=norm(coefsT[k][l])*smooth[l];*(in+l)[1]=0;}

        fftw_execute(p); /* repeat as needed */

        // output can be read into gnuplot
        double sum=0.;
        for (unsigned int l=0;l<coefsT[k].size();l++){
            sum+=(pow(*(in+l)[0],2)+pow(*(in+l)[1],2));
            if((l+1)%nOmega==0){
                fftStream<<sum<<endl;
                sum=0.;
            }
        }
        fftStream<<endl;

    }
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
#endif
}

TimePropagatorOutput& TimePropagatorOutput::withChannelsSubregion(ChannelsSubregion* Channels){
    _channels=Channels;
    return *this;
}
