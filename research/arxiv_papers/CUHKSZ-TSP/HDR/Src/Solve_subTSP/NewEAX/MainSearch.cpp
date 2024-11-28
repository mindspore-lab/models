/*
 * main.cpp
 *   created on: April 24, 2013
 * last updated: May 10, 2020
 *       author: Shujia Liu
 */

#ifndef __ENVIRONMENT__
#include "environment.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include "MainSearch.h"
using namespace std;

Type_dis MainSearch(int Sub_TSP_City_Num, int **Temp_Distance, std::string outfilename){
	InitURandom(); 
	int maxNumOfTrial;
	int maxIte;
	Type_dis finalvalue;

	TEnvironment* gEnv = new TEnvironment();
//	gEnv->fFileNameTSP=(char*)malloc(100);

	/*
	scanf("%d", &maxNumOfTrial);
	scanf("%s", dstFile);
	scanf("%d", &gEnv->fNumOfPop);
	scanf("%d", &gEnv->fNumOfKids);
	*/
//	vector<string> exampleTspMaps{
//		"../tc/eil101.tsp",
//		"../tc/att532.tsp",
//		"../tc/rat575.tsp",
//		"../tc/fnl4461.tsp",
//		"../tc/ja9847.tsp",
//	};
//
//	int id = -1;
//	do {
//		cout << "Please type in 0~4 to choose a dataset." << endl;
//		for (int i = 0; i < exampleTspMaps.size(); i++) {
//			cout << i << ": " << exampleTspMaps[i] << endl;
//		}
//		scanf("%d", &id);
//	} while (id < 0 || id > 4);
//
//	strcpy(gEnv->fFileNameTSP, exampleTspMaps[id].c_str());
// 	scanf("%s", gEnv->fFileNameTSP);
	
	maxNumOfTrial = 1; 	// repeated times
	if(Sub_TSP_City_Num<600){
		gEnv->Npop = 70; 	// number of items
		gEnv->Nch = 40; 	// number of offsprings
		maxIte=300;
	}
	else if(Sub_TSP_City_Num<1000){
		gEnv->Npop = 100; 	// number of items
		gEnv->Nch = 30; 	// number of offsprings
		maxIte=300;
	}
	else{
		gEnv->Npop = 300; 	// number of items
		gEnv->Nch = 30; 	// number of offsprings
		maxIte=300;
	}
//	maxIte=300;
//	gEnv->Npop = 70; 	// number of items
//	gEnv->Nch = 25; 	// number of offsprings

	// cout << "Initializing ..." << endl;
	gEnv->define(Sub_TSP_City_Num, Temp_Distance);
	// cout << "Building solution ..." << endl;

	if(Sub_TSP_City_Num<1000){
		for (int n = 0; n < maxNumOfTrial; ++n){
			gEnv->doIt(maxIte);
	//		gEnv->printOn(n);
			gEnv->writeBest(outfilename);
			finalvalue=gEnv->addwrite(Temp_Distance);
		}

	}
	else {
		gEnv->olddoIt();
		gEnv->writeBest(outfilename);
		finalvalue=gEnv->addwrite(Temp_Distance);

	}

	delete tRand;
	delete gEnv;
	return finalvalue;
}


std::vector<int> SubSearch(int Sub_TSP_City_Num, int **Temp_Distance, int &finalv, int seed){

//	InitURandom();
	InitURandom(seed);
	int maxNumOfTrial;
	int maxIte;
	Type_dis finalvalue;
	std::vector<int> sol;

	TEnvironment* gEnv = new TEnvironment();


	maxNumOfTrial = 1; 	// repeated times
	if(Sub_TSP_City_Num<600){
		gEnv->Npop = 70; 	// number of items
		gEnv->Nch = 40; 	// number of offsprings
		maxIte=300;
	}
	else if(Sub_TSP_City_Num<1000){
		gEnv->Npop = 100; 	// number of items
		gEnv->Nch = 30; 	// number of offsprings
		maxIte=300;
	}
	else{
		gEnv->Npop = 300; 	// number of items
		gEnv->Nch = 30; 	// number of offsprings
		maxIte=300;
	}

	cout << "Initializing ..." << endl;
	gEnv->define(Sub_TSP_City_Num, Temp_Distance);
	cout << "Building solution ..." << endl;

	if(Sub_TSP_City_Num<1000){
		for (int n = 0; n < maxNumOfTrial; ++n){
			gEnv->doIt(maxIte);
	//		gEnv->printOn(n);
//			gEnv->writeBest();
			gEnv->writeBestCopy(sol);
			finalvalue=gEnv->addwrite(Temp_Distance);
		}

	}
	else {
		gEnv->olddoIt();
		gEnv->writeBestCopy(sol);
		finalvalue=gEnv->addwrite(Temp_Distance);

	}
	finalv=finalvalue;
	delete tRand;
	delete gEnv;

	return sol;

}


Type_dis NewSubSearch(int Sub_TSP_City_Num, int **Temp_Distance, int *finalSol){
	InitURandom();
	int maxNumOfTrial;
	int maxIte;
	Type_dis finalvalue;

	TEnvironment* gEnv = new TEnvironment();
//	gEnv->fFileNameTSP=(char*)malloc(100);

	maxNumOfTrial = 1; 	// repeated times
	if(Sub_TSP_City_Num<600){
		gEnv->Npop = 70; 	// number of items
		gEnv->Nch = 40; 	// number of offsprings
		maxIte=300;
	}
	else if(Sub_TSP_City_Num<1000){
		gEnv->Npop = 100; 	// number of items
		gEnv->Nch = 30; 	// number of offsprings
		maxIte=300;
	}
	else{
		gEnv->Npop = 300; 	// number of items
		gEnv->Nch = 30; 	// number of offsprings
		maxIte=300;
	}
//	maxIte=300;
//	gEnv->Npop = 70; 	// number of items
//	gEnv->Nch = 25; 	// number of offsprings

	// cout << "Initializing ..." << endl;
	gEnv->define(Sub_TSP_City_Num, Temp_Distance);
	// cout << "Building solution ..." << endl;

	if(Sub_TSP_City_Num<1000){
		for (int n = 0; n < maxNumOfTrial; ++n){
			gEnv->doIt(maxIte);
	//		gEnv->printOn(n);
//			gEnv->writeBest(outfilename);
			gEnv->write2array(finalSol);
			finalvalue=gEnv->addwrite(Temp_Distance);
		}

	}
	else {
		gEnv->olddoIt();
//		gEnv->writeBest(outfilename);
		finalvalue=gEnv->addwrite(Temp_Distance);

	}

	delete tRand;
	delete gEnv;
	return finalvalue;
}
