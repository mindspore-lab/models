/*
 * Cmdline.h
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#ifndef OBJECT_NEWCMDLINE_H_
#define OBJECT_NEWCMDLINE_H_
#include <string>

class NewCmdline {
public:
	int seed;

	std::string insname;
	std::string edgename;

	std::string roottoIns;
	std::string roottoSol;
	std::string pathtoIns;
	std::string pathtoSol;
	std::string pathtoEdge;
	std::string rep;

	NewCmdline(int count, char *arguments[]);
	void display_help(void);
	void layoutInfo(double avgv, double endtime);
};

#endif /* OBJECT_CMDLINE_H_ */
