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

int main(int argc, char *argv[]) {

	TransTSPLIB(argc, argv);
	return 0;
}
