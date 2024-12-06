/*
 * Cmdline.h
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#ifndef OBJECT_CMDLINE_H_
#define OBJECT_CMDLINE_H_
#include <string>

class Cmdline {
public:
	int seed;
	std::string pathtoIns;
	std::string pathtoSol;
	std::string pathtoPro;
	std::string insname;
	std::string rep;

	Cmdline(int count, char *arguments[]);
	void display_help(void);
	void layoutInfo(double avgv, double endtime);
};

#endif /* OBJECT_CMDLINE_H_ */
