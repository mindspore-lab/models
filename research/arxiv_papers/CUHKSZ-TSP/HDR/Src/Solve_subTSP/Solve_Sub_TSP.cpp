//============================================================================
// Name        : Solve_Sub_TSP.cpp
// Author      : Jintong REN
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "Solve_Sub_TSP.h"
#include "NewEAX/MainSearch.h"
#include <iostream>

typedef double Cost_Type;
typedef int Distance_Type;

//Distance_Type Solve_Sub_TSP(int Sub_TSP_City_Num, int **Temp_Distance, int **Temp_Prab_To_Select,std::string outfilename){
//	Distance_Type bst_dis=.0;
//
//	bst_dis=MainSearch(Sub_TSP_City_Num, Temp_Distance, outfilename);
//
//	return bst_dis;
//}

std::vector<int> Solve_ST(int Sub_TSP_City_Num, int **Temp_Distance, int &finalv, int seed){
	return SubSearch(Sub_TSP_City_Num, Temp_Distance, finalv, seed);
}


Distance_Type Solve_Sub_TSP(int Sub_TSP_City_Num, int **Temp_Distance, int *finalSol){
	Distance_Type bst_dis=.0;

	bst_dis=NewSubSearch(Sub_TSP_City_Num, Temp_Distance, finalSol);

	return bst_dis;
}
