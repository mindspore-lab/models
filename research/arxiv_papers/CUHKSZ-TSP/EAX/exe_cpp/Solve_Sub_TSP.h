/*
 * Solve_Sub_TSP.h
 *
 *  Created on: Jan 21, 2022
 *      Author: jiren
 */

#ifndef SOLVE_SUB_TSP_H_
#define SOLVE_SUB_TSP_H_

#include <vector>

int Solve_Sub_TSP(int Sub_TSP_City_Num, int **Temp_Distance, int **Temp_Prab_To_Select);

std::vector<int> Solve_ST(int Sub_TSP_City_Num, int **Temp_Distance, int &finalv, int seed);

std::vector<int> Solve_ST_Third(int Sub_TSP_City_Num, const std::vector<double> &x, const std::vector<double> &y,
		const std::vector<std::vector<int>> &adjedge ,long int &finalv, int seed);

#endif /* SOLVE_SUB_TSP_H_ */
