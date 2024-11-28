/*
 * Cmdline.h
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#ifndef OBJECT_THIRDNEWCMDLINE_H_
#define OBJECT_THIRDNEWCMDLINE_H_
#include <string>
#include <vector>

class ThirdNewCmdline {
public:
	int seed;

	std::string insname;
	std::string edgename;

	std::string roottoIns;
	std::string roottoSol;
	std::string pathtoIns;
	std::string pathtoSol;
	std::string pathtoEdge;
	std::string pathtoDetail;
	std::string rep;

	ThirdNewCmdline(int count, char *arguments[]);
	void display_help(void);
	void layoutInfo(double avgv, double endtime);
	void layoutDetail(long int value, const std::vector<int> &sol);
};

#endif /* OBJECT_CMDLINE_H_ */
