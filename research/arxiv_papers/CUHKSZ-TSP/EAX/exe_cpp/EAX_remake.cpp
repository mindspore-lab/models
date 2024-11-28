//============================================================================
// Name        : EAX_remake.cpp
// Author      : Jintong
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <string>
#include "Object/Cmdline.h"
#include "Object/Param.h"
#include "Object/NewCmdline.h"
#include "Object/NewParam.h"
#include "Object/ThirdNewCmdline.h"
#include "Object/ThirdNewParam.h"
#include "NewEAX/MainSearch.h"
#include "Solve_Sub_TSP.h"


double get_time(clock_t glo_start){
	double endTimeSeconds=0.0;
	endTimeSeconds = (clock()-glo_start)*1.0/CLOCKS_PER_SEC;
	return (endTimeSeconds);
}

void TransTSPLIB(int argc, char *argv[]){

	try{

		ThirdNewCmdline newcmdl(argc, argv);
		SubSearch_Forth(newcmdl.pathtoIns, newcmdl.pathtoDetail, newcmdl.pathtoSol, newcmdl.seed);

	}
	catch (const std::string& e) { std::cout << "EXCEPTION | " << e << std::endl; }
	catch (const std::exception& e) { std::cout << "EXCEPTION | " << e.what() << std::endl; }
}


//void TransCoord(int argc, char *argv[]){
//	clock_t glo_start;
//
//	try{
//
//		int numNode=1;
//
////		double avgdis;
//		long int finalv;
//		/*************************************************************/
//
//		ThirdNewCmdline newcmdl(argc, argv);
//		ThirdNewParam *Newparamap=new ThirdNewParam(newcmdl.seed, newcmdl.pathtoIns, newcmdl.pathtoEdge);
//		numNode=Newparamap->numNode;
//		Newparamap->layout_map();
//	/*************************************************************/
//		std::cout<<numNode<<std::endl;
//		glo_start=clock();
//		std::vector<int> sol;
//
////		long long int d=0;
////		for (int i=0;i<numNode-1;++i) d+= Newparamap->fetchDis(i, i+1);
////		d+= Newparamap->fetchDis(0, numNode-1);
////		std::cout<<d<<std::endl;
////		exit(0);
//
//		sol=Solve_ST_Third(Newparamap->numNode, Newparamap->x,Newparamap->y, Newparamap->adjEdge, finalv, newcmdl.seed);
////		std::cout<<"sol size:"<<sol.size()<<std::endl;
////		for(int i=0;i<numNode;++i) std::cout<<sol[i]<<" ";
////		std::cout<<std::endl;
//
//
//		if(Newparamap->judgeFeasible(sol) && Newparamap->judgeExistence(sol)) newcmdl.layoutInfo(1.0*finalv, get_time(glo_start));
//		else newcmdl.layoutInfo(0, get_time(glo_start));
//
//		std::cout.setf(std::ios::fixed);
//		std::cout<<"AvgCost="<<finalv<<std::endl;
//		std::cout<<"Endtime="<<get_time(glo_start)<<std::endl;
//
//		newcmdl.layoutDetail(finalv, sol);
//
//		/*************************************************************/
//
//		delete Newparamap;
//		/*************************************************************/
//
//	}
//	catch (const std::string& e) { std::cout << "EXCEPTION | " << e << std::endl; }
//	catch (const std::exception& e) { std::cout << "EXCEPTION | " << e.what() << std::endl; }
//}

int main(int argc, char *argv[]) {

//	TransCoord(argc, argv);
	TransTSPLIB(argc, argv);
	return 0;
}
