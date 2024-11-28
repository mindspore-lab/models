/*
 * Param.h
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#ifndef OBJECT_THIRDNEWPARAM_H_
#define OBJECT_THIRDNEWPARAM_H_
#include <vector>
#include <string>

//typedef long long int TyINT;
typedef int TyINT;
#define PI 3.14159265358979323846264


class ThirdNewParam{
public:
	int numNode;
//	int **MatrixDis;

	std::vector<double> x;
	std::vector<double> y;

	std::vector<std::vector<int>> adjEdge;

	ThirdNewParam(int seed, std::string pathtoIns, std::string pathtoEdge);
	~ThirdNewParam();
	TyINT fetchDis(int n1, int n2);
	bool checkexist(int base, int n);
	bool judgeFeasible(const std::vector<int> &sol);
	bool judgeExistence(const std::vector<int> &sol);
//	bool judgeCyclic(const std::vector<int> &sol);
	void layout_map();
};

#endif /* OBJECT_PARAM_H_ */
