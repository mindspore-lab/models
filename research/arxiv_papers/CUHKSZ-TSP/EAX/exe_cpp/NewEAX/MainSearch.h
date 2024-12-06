#ifndef NEWEAX_MAINSEARCH_H_
#define NEWEAX_MAINSEARCH_H_

#include <vector>
#include <string>
std::vector<int> SubSearch_Third(int Sub_TSP_City_Num, const std::vector<double> &x, const std::vector<double> &y,
		const std::vector<std::vector<int>> &adjedge ,long int &finalv, int seed);

void SubSearch_Forth(std::string fileN, std::string fileR, std::string fileS, int seed);

#endif
