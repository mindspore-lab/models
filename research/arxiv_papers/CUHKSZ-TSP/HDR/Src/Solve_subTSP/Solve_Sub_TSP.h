/*
 * Solve_Sub_TSP.h
 *
 *  Created on: Jan 21, 2022
 *      Author: jiren
 */

#ifndef SOLVE_SUB_TSP_H_
#define SOLVE_SUB_TSP_H_

#include <vector>
#include <string>

//int Solve_Sub_TSP(int Sub_TSP_City_Num, int **Temp_Distance, int **Temp_Prab_To_Select, std::string outfilename);

std::vector<int> Solve_ST(int Sub_TSP_City_Num, int **Temp_Distance, int &finalv, int seed);

int Solve_Sub_TSP(int Sub_TSP_City_Num, int **Temp_Distance, int *finalSol);

#endif /* SOLVE_SUB_TSP_H_ */
