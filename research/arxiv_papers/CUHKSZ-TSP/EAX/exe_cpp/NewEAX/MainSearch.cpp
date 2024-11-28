#ifndef __ENVIRONMENT__
#include "env.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <vector>

void SubSearch_Forth(std::string fileN, std::string fileR, std::string fileS,  int seed){
	  int maxNumOfTrial;

	  char* dstFile = const_cast<char*> (fileR.c_str());

	  TEnvironment* gEnv = NULL;
	  gEnv = new TEnvironment();

	  maxNumOfTrial=1;
	  InitURandom(seed);
	  gEnv->fNumOfPop = 300;
	  gEnv->fNumOfKids = 30;
	  gEnv->fFileNameTSP =  const_cast<char*> (fileN.c_str());
	  gEnv->fFileNameInitPop = NULL;
//	  if( argc == 7 )
//	    gEnv->fFileNameInitPop = argv[6];
	  bool tflag;
	  tflag=gEnv->Define();

	  if(!tflag){
		  delete gEnv;
		  return;
	  }

	  for( int n = 0; n < maxNumOfTrial; ++n )
	  {
	    gEnv->DoIt();

	    gEnv->PrintOn( n, dstFile );
	    gEnv->WriteBest( dstFile, fileS );
	    // gEnv->WritePop( n, dstFile );
	  }


	  delete gEnv;
}


//std::vector<int> SubSearch_Third(int Sub_TSP_City_Num, const std::vector<double> &x, const std::vector<double> &y,
//		const std::vector<std::vector<int>> &adjedge ,long int &finalv, int seed){
////	char* dstFile = argv[2];
//	std::vector<int> sol(Sub_TSP_City_Num);
//	bool flag_r;
//	TEnvironment* gEnv = NULL;
//	gEnv = new TEnvironment();
//	InitURandom(seed);
//	gEnv->fNumOfPop=300;
//	gEnv->fNumOfKids=30;
//
////	gEnv->Define();
//	printf("start to initialize:\n");
//	flag_r=gEnv->ThirdDefine(Sub_TSP_City_Num, x, y, adjedge);
//
//	if(!flag_r){
//		delete gEnv;
//		return sol;
//	}
//
//	printf("Building:\n");
//	gEnv->DoIt();
//
////	gEnv->PrintOn( n, dstFile );
////	gEnv->WriteBest( dstFile );
//
//	gEnv->FetchSol(sol, finalv);
//
//	delete gEnv;
//	return sol;
//}


//int main( int argc, char* argv[] )
//{
//  int maxNumOfTrial;
//
//  sscanf( argv[1], "%d", &maxNumOfTrial );
//  char* dstFile = argv[2];
//
//  TEnvironment* gEnv = NULL;
//  gEnv = new TEnvironment();
//  InitURandom();
//
//  int d;
//  sscanf( argv[3], "%d", &d );
//  gEnv->fNumOfPop = d;
//  sscanf( argv[4], "%d", &d );
//  gEnv->fNumOfKids = d;
//  gEnv->fFileNameTSP = argv[5];
//  gEnv->fFileNameInitPop = NULL;
//  if( argc == 7 )
//    gEnv->fFileNameInitPop = argv[6];
//
//  gEnv->Define();
//
//  for( int n = 0; n < maxNumOfTrial; ++n )
//  {
//    gEnv->DoIt();
//
//    gEnv->PrintOn( n, dstFile );
//    gEnv->WriteBest( dstFile );
//    // gEnv->WritePop( n, dstFile );
//  }
//
//  return 0;
//}
