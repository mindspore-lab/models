/*
 * Param.cpp
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#include "ThirdNewParam.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>

ThirdNewParam::ThirdNewParam(int seed, std::string pathtoIns, std::string pathtoEdge) {
	// TODO Auto-generated constructor stub
	srand(seed);


	int tpdis=-1;
	double a,b;

	/***********************************/
	std::ifstream inputFile(pathtoIns, std::ios::in);
	while(!inputFile.eof()){
		inputFile>>tpdis>>a>>b;
//		std::cout<<tpdis<<" "<<a<<" "<<b<<std::endl;
		x.push_back(a);
		y.push_back(b);
		inputFile.get();
		if(inputFile.peek()=='\n') break;
	}
	inputFile.close();
	numNode=(int) x.size();

	/***********************************/

	int n1,n2;
	adjEdge=std::vector<std::vector<int>> (numNode);
	std::ifstream inputFile2(pathtoEdge, std::ios::in);
	while(!inputFile2.eof()){
		inputFile2>>n1>>n2;
//		std::cout<<n1<<" "<<n2<<std::endl;
		adjEdge[n1].push_back(n2);
		adjEdge[n2].push_back(n1);
		inputFile2.get();
		if(inputFile2.peek()=='\n') break;
	}
	inputFile2.close();

	/************************************/

//	std::cout<<fetchDis(0,1);
}

ThirdNewParam::~ThirdNewParam(){

}

TyINT ThirdNewParam::fetchDis(int n1, int n2){

	if(checkexist(n1, n2)) return 0;

	double lati, latj, longi, longj;
	double q1, q2, q3, q4, q5;

	lati = x[n1] * PI / 180.0;
	latj = x[n2] * PI / 180.0;

	longi = y[n1] * PI / 180.0;
	longj = y[n2] * PI / 180.0;

	q1 = cos(latj) * sin(longi - longj);
	q3 = sin((longi - longj) / 2.0);
	q4 = cos((longi - longj) / 2.0);
	q2 = sin(lati + latj) * q3 * q3 - sin(lati - latj) * q4 * q4;
	q5 = cos(lati - latj) * q4 * q4 - cos(lati + latj) * q3 * q3;
	return (TyINT)(6378388.0 * atan2(sqrt(q1 * q1 + q2 * q2), q5) + 1.0);
}

bool ThirdNewParam::checkexist(int base, int n){
	for(unsigned int i=0;i<adjEdge[base].size();++i)
		if(n==adjEdge[base][i]) return true;
	return false;
}

bool ThirdNewParam::judgeFeasible(const std::vector<int> &sol){

	int ct=0;
	std::vector<std::vector<int>> solv(numNode);
	for(int i=0;i<numNode;++i) solv[i].resize(2);
//	int **solv=new int *[numNode];
//	for(int i=0;i<numNode;++i) solv[i]=new int [2];


	for(int i=1;i<numNode-1;++i){
		solv[sol[i]][0]=sol[i-1];
		solv[sol[i]][1]=sol[i+1];
	}
	solv[sol[0]][0]=sol[numNode-1];
	solv[sol[0]][1]=sol[1];
	solv[sol[numNode-1]][0]=sol[numNode-2];
	solv[sol[numNode-1]][1]=sol[0];

	for(int i=0;i<numNode;++i){
		for(unsigned int j=0;j<adjEdge[i].size();++j){
			if(adjEdge[i][j]!=solv[i][0] && adjEdge[i][j]!=solv[i][1]) ct+=1;
		}
	}

	if(ct>0) return false;
	return true;
//	return Flag;
}

bool ThirdNewParam::judgeExistence(const std::vector<int> &sol){
	std::vector<bool> Fv(numNode, false);

	for(int i=0;i<numNode;++i){
		if(Fv[sol[i]]) return false;
		Fv[sol[i]] = true;
	}
	return true;
}

//bool NewParam::judgeCyclic(const std::vector<int> &sol){
//	std::vector<bool> Fv(numNode, false);
//
//
//}

void ThirdNewParam::layout_map(void){

//	for(int i=0;i<numNode;++i){
//		std::cout<<i<<": ";
//		for(unsigned int j=0;j<adjEdge[i].size();++j) std::cout<<adjEdge[i][j]<<" ";
//		std::cout<<std::endl;
//	}
//
//	for(int i=0;i<numNode;i++){
//
//		std::cout<<i<<": ";
//
//		for(int j=0;j<numNode;j++) std::cout<<MatrixDis[i][j]<<" ";
//		std::cout<<std::endl;
//	}

}
