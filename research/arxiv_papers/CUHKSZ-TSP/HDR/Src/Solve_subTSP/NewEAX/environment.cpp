/*
 * environment.cpp
 *   created on: April 24, 2013
 * last updated: May 10, 2020
 *       author: Shujia Liu
 */

#ifndef __ENVIRONMENT__
#include "environment.h"
#endif
#include <math.h>
#include <iostream>
using namespace std;

//void MakeRandSol(TEvaluator* eval, TIndi& indi);
//void Make2optSol(TEvaluator* eval, TIndi& indi);

TEnvironment::TEnvironment(){
	fEvaluator = new TEvaluator();

	/*others*/
	fFileNameTSP=NULL;

	Npop=0; // the number of population
	Nch=0; // the number of children generated from one pair of parent

	fCurNumOfGen=0; // the number of generations of the current population
	fAccumurateNumCh=0; // accumulated number of children so far

	fBestNumOfGen=0; // the number of generations that generates the current best solution
	fBestAccumeratedNumCh=0; // accumulated number of generations of the current best solution

	fAverageValue=0; // average road length of TSP in a population
	fBestValue=0; // road length of the best solution in a population
	fBestIndex=0;	// index of the best solution in a population

	fStagBest=0; // accumulated number of generations that doesn't generate a better solution compared to previous generation
	fStage=0; // the current step of genetic algorithm
	fMaxStagBest=0; // the genetic algorithm goes into the next stage when fStagBest == fMaxStagBest
	fCurNumOfGen1=0; // the number of generations when stage 1 completes

	tKopt=NULL;
	tCross=NULL;
	tCurPop=NULL;

	fTimeStart=clock();
	fTimeInit=clock();
	fTimeEnd=clock();
}

TEnvironment::~TEnvironment(){
	delete [] tCurPop;
	delete fEvaluator;

	delete tCross;
	delete tKopt;
}

void TEnvironment::define(int Sub_TSP_City_Num, int **Temp_Distance){
//	fEvaluator->setInstance( fFileNameTSP );
	fEvaluator->setMatrix(Sub_TSP_City_Num, Temp_Distance);
	int N = fEvaluator->Ncity;
	fIndexForMating.resize(Npop + 1);

	tCurPop = new TIndi [ Npop ];
	for ( int i = 0; i < Npop; ++i ) {
		tCurPop[i].define( N );
	}
	tBest.define( N );
	tCross = new TCross( N );
	tCross->eval = fEvaluator;
	tCross->Npop = Npop;


	tKopt = new TKopt( N );
	tKopt->eval = fEvaluator;
	tKopt->setInvNearList();

	fEdgeFreq.clear();
	for (int i = 0; i < N; i++) {
		vector<int> row(N);
		fEdgeFreq.push_back(row);
	}
}

void TEnvironment::doIt(int maxIte){
	this->fTimeStart = clock();
	this->initPop(); // initializes population
	this->fTimeInit = clock();
	this->init();
	int ct=0;

	this->getEdgeFreq();
//	while( 1 ){
	while( ct<=maxIte ){
		this->setAverageBest();
//		printf( "%d:\t%ld\t%lf\n", fCurNumOfGen, fBestValue, fAverageValue );
		// cout<<fCurNumOfGen<<":\t"<<fBestValue+fEvaluator->fixcplus<<"\t"<<fAverageValue+fEvaluator->fixcplus<<endl;
		if( this->terminationCondition() ) break;

		this->selectForMating();
		for( int s =0; s < Npop; ++s ) this->generateKids( s );

		++fCurNumOfGen;
		++ct;
	}
	this->fTimeEnd = clock();
}

void TEnvironment::olddoIt(){
	this->fTimeStart = clock();
	this->initPop(); // initializes population
	this->fTimeInit = clock();
	this->init();
	int ct=0;

	this->getEdgeFreq();
	while( 1 ){
//	while( ct<=10 ){
		this->setAverageBest();
//		printf( "%d:\t%ld\t%lf\n", fCurNumOfGen, fBestValue, fAverageValue );
		// cout<<fCurNumOfGen<<":\t"<<fBestValue+fEvaluator->fixcplus<<"\t"<<fAverageValue+fEvaluator->fixcplus<<endl;
		if( this->terminationCondition() ) break;

		this->selectForMating();
		for( int s =0; s < Npop; ++s ) this->generateKids( s );

		++fCurNumOfGen;
		++ct;
	}
	this->fTimeEnd = clock();
}

void TEnvironment::init(){
	fAccumurateNumCh = 0;
	fCurNumOfGen = 0;
	fStagBest = 0;
	fMaxStagBest = 0;
	fStage = 1; // sets stage to 1
	fFlagC[ 0 ] = 4; // maintains population diversity	1:Greedy, 2:---, 3:Distance, 4:Entropy
	fFlagC[ 1 ] = 1; // the type of Eset: 1:Single-AB, 2:Block2
}

bool TEnvironment::terminationCondition(){
	if ( fAverageValue - fBestValue < 0.001 )  return true;
	if( fStage == 1 ){
		if( fStagBest == int(1500/Nch) && fMaxStagBest == 0 ) // 1500/Nch
			fMaxStagBest =int( fCurNumOfGen / 10 ); // fMaxStagBest = G/10
		else if( fMaxStagBest != 0 && fMaxStagBest <= fStagBest ){
			fStagBest = 0;
			fMaxStagBest = 0;
			fCurNumOfGen1 = fCurNumOfGen;
			fFlagC[ 1 ] = 2;
			fStage = 2;
		}
		return false;
	}
	if( fStage == 2 ){
		if( fStagBest == int(1500/Nch) && fMaxStagBest == 0 )			// 1500/Nch
			fMaxStagBest = int( (fCurNumOfGen - fCurNumOfGen1) / 10 );	// fMaxStagBest = G/10
		else if( fMaxStagBest != 0 && fMaxStagBest <= fStagBest ) return true;
		return false;
	}

	return true;
}

void TEnvironment::setAverageBest(){
	Type_dis stockBest = tBest.fEvaluationValue;
	fAverageValue = 0.0;
	fBestIndex = 0;
	fBestValue = tCurPop[0].fEvaluationValue;
	for(int i = 0; i < Npop; ++i ){
		fAverageValue += tCurPop[i].fEvaluationValue;
		if( tCurPop[i].fEvaluationValue < fBestValue ){
			fBestIndex = i;
			fBestValue = tCurPop[i].fEvaluationValue;
		}
	}
	tBest = tCurPop[ fBestIndex ];
	fAverageValue /= (double)Npop;
	if( tBest.fEvaluationValue < stockBest ){
		fStagBest = 0;
		fBestNumOfGen = fCurNumOfGen;
		fBestAccumeratedNumCh = fAccumurateNumCh;
	}
	else ++fStagBest;
}

void TEnvironment::initPop(){
	for ( int i = 0; i < Npop; ++i ){
		tKopt->makeRandSol( tCurPop[ i ] ); // randomly sets a route
		tKopt->doIt( tCurPop[ i ] ); // local search (2-opt neighborhood)
//		std::cout<<i<<" "<<tCurPop[i].fEvaluationValue<<std::endl;
	}
}

void TEnvironment::selectForMating(){
	tRand->permutation( fIndexForMating, Npop, Npop );
	fIndexForMating[ Npop ] = fIndexForMating[ 0 ];
}

void TEnvironment::generateKids( int s ){
	// tCurPop[fIndexForMating[s]] gets replaced by the best solution in tCross->DoIt()
	// fEdgeFreq[][] ͬʱ������
	tCross->setParents( tCurPop[fIndexForMating[s]], tCurPop[fIndexForMating[s+1]], fFlagC, Nch );
	tCross->doIt( tCurPop[fIndexForMating[s]], tCurPop[fIndexForMating[s+1]], Nch, 1, fFlagC, fEdgeFreq );
	fAccumurateNumCh += tCross->fNumOfGeneratedCh;
}

void TEnvironment::getEdgeFreq(){
	int  k0, k1, N = fEvaluator->Ncity;
//	std::cout << fEdgeFreq.size() << " " << fEdgeFreq[0].size() << std::endl;
	for( int j1 = 0; j1 < N; ++j1 )
		for( int j2 = 0; j2 < N; ++j2 )
			fEdgeFreq[ j1 ][ j2 ] = 0;

	for( int i = 0; i < Npop; ++i )
		for(int j = 0; j < N; ++j ){
			k0 = tCurPop[ i ].fLink[ j ][ 0 ];
			k1 = tCurPop[ i ].fLink[ j ][ 1 ];
			++fEdgeFreq[ j ][ k0 ];
			++fEdgeFreq[ j ][ k1 ];
		}
}

void TEnvironment::printOn( int n ){
	printf( "n = %d val = %lld Gen = %d Time = %d %d\n" , n, tBest.fEvaluationValue, fCurNumOfGen,
		(int)((double)(this->fTimeInit - this->fTimeStart)/(double)CLOCKS_PER_SEC),
		(int)((double)(this->fTimeEnd - this->fTimeStart)/(double)CLOCKS_PER_SEC) );
	fflush(stdout);

}

void TEnvironment::writeBest(std::string outfilename){
	FILE *fp;
	char filename[ 80 ];

	sprintf( filename, outfilename.c_str() );
//	sprintf( filename, "NoPerturb_Best_Solution.tsp" );
	fp = fopen( filename, "w");
	fEvaluator->writeTo( fp, tBest);
	fclose( fp );
}

void TEnvironment::writeBestCopy(std::vector<int> &sol){
	FILE *fp;
	char filename[ 80 ];

	sprintf( filename, "NoPerturb_Best_Solution.tsp" );
	fp = fopen( filename, "w");
	fEvaluator->writeToCopy( fp, tBest, sol);
	fclose( fp );

}

void TEnvironment::write2array(int *finalSol){
	fEvaluator->write2Array( tBest, finalSol);
}

Type_dis TEnvironment::addwrite(int **Mdis){
	Type_dis dis=fEvaluator->AnodoIt(tBest, Mdis);
//	Type_dis dis=fEvaluator->SecdiIt(tBest, Mdis);
//	std::cout<<"Real Dis:"<<dis<<std::endl;
//	fEvaluator->doIt(tBest);
//	std::cout<<"Inside Dis:"<<tBest.fEvaluationValue<<std::endl;
	return dis;


}
