/*
 * Param.h
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#ifndef OBJECT_PARAM_H_
#define OBJECT_PARAM_H_
#include <vector>
#include <string>

class Param{
public:
	int numNode;
	int **MatrixDis;
	int **Prob;
//	std::vector<std::vector<int>> MatrixDis;

	std::vector<std::vector<int>> adjEdge;

	Param(int numN, int seed, std::string pathtoIns, std::string pathtoPro);
	~Param();
	bool judgeFeasible(const std::vector<int> &sol);
	bool judgeExistence(const std::vector<int> &sol);
	void layout_map();
};

#endif /* OBJECT_PARAM_H_ */
