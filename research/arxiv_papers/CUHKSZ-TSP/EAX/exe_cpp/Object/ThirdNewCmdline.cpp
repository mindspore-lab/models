/*
 * Cmdline.cpp
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#include "ThirdNewCmdline.h"
#include <iostream>
#include <fstream>

ThirdNewCmdline::ThirdNewCmdline(int count, char *arguments[]) {
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
	pathtoDetail="Detail/"+insname+"D"+rep;
	std::cout<<"Input instance:"<<pathtoIns<<std::endl;
	std::cout<<"Input edge:"<<pathtoEdge<<std::endl;
	std::cout<<"Output instance:"<<pathtoSol<<std::endl;
	std::cout<<"Detail Sol:"<<pathtoDetail<<std::endl;
}

void ThirdNewCmdline::display_help(void){
	std::cout << std::endl;
	std::cout << "-------------------------------------------------- EAX-TSP-remake algorithm (2022) --------------------------------------------------" << std::endl;
	std::cout << "-------------------------------------------------------------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;
}

void ThirdNewCmdline::layoutInfo(double avgv, double endtime){
	std::ofstream outf(pathtoSol);
	if(outf.is_open()){
		outf.setf(std::ios::fixed);
		outf<<"1 "<<avgv<<" "<<endtime<<std::endl;
		outf.close();
	}
}

void ThirdNewCmdline::layoutDetail(long int value, const std::vector<int> &sol){

	std::ofstream outf(pathtoDetail);
	if(outf.is_open()){
		outf.setf(std::ios::fixed);
		outf<<value<<std::endl<<std::endl;

		for(unsigned int i=0;i<sol.size();++i) outf<<sol[i]<<std::endl;
		outf.close();
	}
}
