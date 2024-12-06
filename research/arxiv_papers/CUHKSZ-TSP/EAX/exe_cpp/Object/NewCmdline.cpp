/*
 * Cmdline.cpp
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#include "NewCmdline.h"
#include <iostream>
#include <fstream>

NewCmdline::NewCmdline(int count, char *arguments[]) {
	while (count != 1) {

		if (std::string(arguments[count - 2]) == "-i") insname=std::string(arguments[count - 1]);
		else
			if (std::string(arguments[count - 2]) =="-e") edgename= std::string(arguments[count - 1]);
		else
			if (std::string(arguments[count - 2]) =="--seed") seed= atoi(arguments[count - 1]);
		else
			if (std::string(arguments[count - 2]) =="-rep") rep=std::string(arguments[count - 1]);
		else
			if (std::string(arguments[count - 2]) =="-s") roottoIns=std::string(arguments[count - 1]);	//path to layout the best found solution
		else
			if (std::string(arguments[count - 2]) =="-d") roottoSol=std::string(arguments[count - 1]);	//path to layout the best found solution
		else {  // unknow parameter
			std::cout<<""<<std::endl;
			display_help();
			throw std::string("incorrect input");
		}
		count = count - 2;
	}

	pathtoIns=roottoIns+"/"+insname;
	pathtoEdge=roottoIns+"/"+edgename;
	pathtoSol=roottoSol+"/"+insname+"F"+rep;
	std::cout<<"Input instance:"<<pathtoIns<<std::endl;
	std::cout<<"Input edge:"<<pathtoEdge<<std::endl;
	std::cout<<"Output instance:"<<pathtoSol<<std::endl;

}

void NewCmdline::display_help(void){
	std::cout << std::endl;
	std::cout << "-------------------------------------------------- EAX-TSP-remake algorithm (2022) --------------------------------------------------" << std::endl;
	std::cout << "-------------------------------------------------------------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;
}

void NewCmdline::layoutInfo(double avgv, double endtime){
	std::ofstream outf(pathtoSol);
	if(outf.is_open()){
		outf.setf(std::ios::fixed);
		outf<<"1 "<<avgv<<" "<<endtime<<std::endl;
		outf.close();
	}
}
