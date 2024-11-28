/*
 * Cmdline.cpp
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#include "Cmdline.h"
#include <iostream>
#include <fstream>

Cmdline::Cmdline(int count, char *arguments[]) {
	while (count != 1) {

		if (std::string(arguments[count - 2]) == "-i") insname=std::string(arguments[count - 1]);
		else
			if (std::string(arguments[count - 2]) =="--seed") seed= atoi(arguments[count - 1]);
		else
			if (std::string(arguments[count - 2]) =="-rep") rep=std::string(arguments[count - 1]);
		else
			if (std::string(arguments[count - 2]) =="-s") pathtoIns=std::string(arguments[count - 1]);	//path to layout the best found solution
		else
			if (std::string(arguments[count - 2]) =="-d") pathtoSol=std::string(arguments[count - 1]);	//path to layout the best found solution
		else
			if (std::string(arguments[count - 2]) =="-p") pathtoPro=std::string(arguments[count - 1]);	//path to layout the best found solution

		else {  // unknow parameter
			std::cout<<""<<std::endl;
			display_help();
			throw std::string("incorrect input");
		}
		count = count - 2;
	}

	pathtoIns=pathtoIns+"/"+insname;
	pathtoSol=pathtoSol+"/"+insname+"F"+rep;
	// std::cout<<"Input instance:"<<pathtoIns<<std::endl;
	// std::cout<<"Output instance:"<<pathtoSol<<std::endl;

}

void Cmdline::display_help(void){
	std::cout << std::endl;
	std::cout << "-------------------------------------------------- EAX-TSP-remake algorithm (2022) --------------------------------------------------" << std::endl;
	std::cout << "-------------------------------------------------------------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;
}

void Cmdline::layoutInfo(double avgv, double endtime){
	std::ofstream outf(pathtoSol);
	if(outf.is_open()){
		outf.setf(std::ios::fixed);
		outf<<"1 "<<avgv<<" "<<endtime<<std::endl;
		outf.close();
	}
}
