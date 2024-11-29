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
	  }


	  delete gEnv;
}