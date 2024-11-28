/*
 * Param.cpp
 *
 *  Created on: Jan 12, 2022
 *      Author: jiren
 */

#include "Param.h"
#include <string.h>
#include <iostream>
#include <fstream>

Param::Param(int numN,int seed, std::string pathtoIns, std::string pathtoPro) {
	// TODO Auto-generated constructor stub
	srand(seed);
	numNode=numN;
//	MatrixDis.resize(numNode);
//	for(int i=0;i<numNode;++i) MatrixDis[i].resize(numNode);

	int tpdis=-1;
	std::ifstream inputFile(pathtoIns);
	if(inputFile.is_open()){
		inputFile>>tpdis;
		numNode=tpdis;

		MatrixDis=new int*[numNode];
		for(int i=0;i<numNode;++i) {
			MatrixDis[i]=new int[numNode];
			memset(MatrixDis[i], 0, numNode*sizeof(int));
		}


		for(int i=0;i<numNode;++i){
			for(int j=0;j<numNode;++j) {
				inputFile>>tpdis;
				MatrixDis[i][j]=tpdis;
			}
		}
	}


	/***********************************************************/
	Prob=new int*[numNode];
	for(int i=0;i<numNode;++i) {
		Prob[i]=new int[numNode];
		memset(Prob[i], 0, numNode*sizeof(int));
	}

//	std::ifstream inputFile2(pathtoPro);
//	if(inputFile2.is_open()){
//
//		for(int i=0;i<numNode;++i){
//			for(int j=0;j<numNode;++j) {
//				inputFile2>>tpdis;
//				Prob[i][j]=tpdis;
//			}
//		}
//	}

	/***********************************************************/
	adjEdge=std::vector<std::vector<int>> (numNode);
//	for(int i=0;i<numNode;++i) adjEdge[i]=std::vector<int> (2);
	for(int i=0;i<numNode;++i)
		for(int j=i+1;j<numNode;++j)
			if(MatrixDis[i][j]<0){
				adjEdge[i].push_back(j);
				adjEdge[j].push_back(i);
			}



}

Param::~Param(){
	for(int i=0;i<numNode;++i) delete []MatrixDis[i];
	delete []MatrixDis;

	for(int i=0;i<numNode;++i) delete []Prob[i];
	delete []Prob;
}

void Param::layout_map(void){
	for(int i=0;i<numNode;i++){
		for(int j=0;j<numNode;j++) std::cout<<MatrixDis[i][j]<<" ";
		std::cout<<std::endl;
	}

//	for(int i=0;i<numNode;i++){
//		for(int j=0;j<numNode;j++) std::cout<<Prob[i][j]<<" ";
//		std::cout<<std::endl;
//	}

}

bool Param::judgeFeasible(const std::vector<int> &sol){

	int ct=0;
	std::vector<std::vector<int>> solv(numNode);
	for(int i=0;i<numNode;++i) solv[i].resize(2);

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

bool Param::judgeExistence(const std::vector<int> &sol){
	std::vector<bool> Fv(numNode, false);

	for(int i=0;i<numNode;++i){
		if(Fv[sol[i]]) return false;
		Fv[sol[i]] = true;
	}
	return true;
}



