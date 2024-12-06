/*
 * MainSearch.h
 *
 *  Created on: Jan 21, 2022
 *      Author: jiren
 */

#ifndef NEWEAX_MAINSEARCH_H_
#define NEWEAX_MAINSEARCH_H_
#include "indi.h"
Type_dis MainSearch(int Sub_TSP_City_Num, int **Temp_Distance, std::string outfilename);

std::vector<int> SubSearch(int Sub_TSP_City_Num, int **Temp_Distance, int &finalv, int seed);

Type_dis NewSubSearch(int Sub_TSP_City_Num, int **Temp_Distance, int *finalSol);

#endif /* NEWEAX_MAINSEARCH_H_ */
