#ifndef __ENVIRONMENT__
#include "env.h"
#endif

#include <math.h> 
     
//void MakeRandSol( TEvaluator* eval , TIndi& indi );
//void Make2optSol( TEvaluator* eval , TIndi& indi );

TEnvironment::TEnvironment()
{
  fEvaluator = new TEvaluator();
  tCross=NULL;
  tKopt=NULL;
  fFileNameTSP=NULL;
  fFileNameInitPop=NULL;

  fNumOfPop=0;
  fNumOfKids=0;

  tCurPop=NULL;
  fCurNumOfGen=0;
  fAccumurateNumCh=0;
  fBestNumOfGen=0;

  fBestAccumeratedNumCh=0;

  fEdgeFreq=NULL;
  fAverageValue=0;
  fBestValue=0;
  fBestIndex=0;
  fIndexForMating=NULL;

  fStagBest=0;

  fStage=0;
  fMaxStagBest=0;
  fCurNumOfGen1=0;

  fTimeStart=clock();
  fTimeInit=clock();
  fTimeEnd=clock();
}


TEnvironment::~TEnvironment()
{
	int N = fEvaluator->Ncity;
  delete [] fIndexForMating;
  delete [] tCurPop;
  delete  fEvaluator;
  delete  tCross;
  delete  tKopt;



  for( int i = 0; i < N; ++i ) 
    delete [] fEdgeFreq[ i ];
  delete [] fEdgeFreq;
}


bool TEnvironment::Define()
{
	bool flag_a;
  flag_a=fEvaluator->SetInstance( fFileNameTSP );
  int N = fEvaluator->Ncity;

  fIndexForMating = new int [ fNumOfPop + 1 ];  

  tCurPop = new TIndi [ fNumOfPop ];
  for ( int i = 0; i < fNumOfPop; ++i )
    tCurPop[i].Define( N );

  tBest.Define( N );

  tCross = new TCross( N );
  tCross->eval = fEvaluator;                 
  tCross->fNumOfPop = fNumOfPop;             

  tKopt = new TKopt( N );
  tKopt->eval = fEvaluator;
  if(flag_a) tKopt->SetInvNearList();

  fEdgeFreq = new int* [ N ]; 
  for( int i = 0; i < N; ++i ) 
    fEdgeFreq[ i ] = new int [ fEvaluator->fNearNumMax+1 ]; // Large


  return flag_a;


}

bool TEnvironment::ThirdDefine(int Sub_TSP_City_Num, const std::vector<double> &x, const std::vector<double> &y,
		const std::vector<std::vector<int>> &adjedge){
	bool flag_accept=false;
	flag_accept=fEvaluator->ThirdSet(Sub_TSP_City_Num, x, y, adjedge);


	  int N = fEvaluator->Ncity;

	  fIndexForMating = new int [ fNumOfPop + 1 ];

	  tCurPop = new TIndi [ fNumOfPop ];
	  for ( int i = 0; i < fNumOfPop; ++i )
	    tCurPop[i].Define( N );

	  tBest.Define( N );

	  tCross = new TCross( N );
	  tCross->eval = fEvaluator;
	  tCross->fNumOfPop = fNumOfPop;

	  tKopt = new TKopt( N );
	  tKopt->eval = fEvaluator;
	  tKopt->SetInvNearList();

	  fEdgeFreq = new int* [ N ];
	  for( int i = 0; i < N; ++i )
	    fEdgeFreq[ i ] = new int [ fEvaluator->fNearNumMax+1 ]; // Large

	  return flag_accept;

}


void TEnvironment::DoIt()
{
//  this->fTimeStart = clock();
  this->fTimeStart = fEvaluator->fTStart;

//  std::cout<<"Time:"<<(clock() - this->fTimeStart)/(double)CLOCKS_PER_SEC<<std::endl;
//
//  exit(0);

  if( fFileNameInitPop == NULL )
    this->InitPop();                       
  else
    this->ReadPop( fFileNameInitPop );     

  std::cout<<"IniTime:"<<(clock() - this->fTimeStart)/(double)CLOCKS_PER_SEC<<std::endl;

 if(((clock() - this->fTimeStart)/(double)CLOCKS_PER_SEC)>fEvaluator->Ncity/5.0) return;
//  if(((clock() - this->fTimeStart)/(double)CLOCKS_PER_SEC)>5.0) return;

  this->fTimeInit = clock();    

  this->Init();
  this->GetEdgeFreq();

  while( 1 )
 {
    this->SetAverageBest();
//    printf( "%d: %d %lf\n", fCurNumOfGen, fBestValue, fAverageValue );
    std::cout<<fCurNumOfGen<<"\t"<<fBestValue+fEvaluator->fixPlus<<"\t"<<fAverageValue+fEvaluator->fixPlus<<\
    		'\t'<<(clock() - this->fTimeStart)/(double)CLOCKS_PER_SEC<<std::endl;

    if( this->TerminationCondition() ) break;

    this->SelectForMating();

    for( int s =0; s < fNumOfPop; ++s )
    {
      this->GenerateKids( s );     
      this->SelectForSurvival( s ); 
    }
    ++fCurNumOfGen;

    if(((clock() - this->fTimeStart)/(double)CLOCKS_PER_SEC)>fEvaluator->Ncity/5.0) break;
  }

  this->fTimeEnd = clock();   
}
 
/* See Section 2.2 */
void TEnvironment::Init()
{
  fAccumurateNumCh = 0;
  fCurNumOfGen = 0;
  fStagBest = 0;
  fMaxStagBest = 0;
  fStage = 1;          /* Stage I */
  fFlagC[ 0 ] = 4;     /* Diversity preservation: 1:Greedy, 2:--- , 3:Distance, 4:Entropy (see Section 4) */
  fFlagC[ 1 ] = 1;     /* Eset Type: 1:Single-AB, 2:Block2 (see Section 3) */ 
} 

/* See Section 2.2 */
bool TEnvironment::TerminationCondition()
{
  if ( fAverageValue - fBestValue < 0.001 )  
    return true;

  if( fStage == 1 ) /* Stage I */      
  {
    if( fStagBest == int(1500/fNumOfKids) && fMaxStagBest == 0 ){ /* 1500/N_ch (See Section 2.2) */
      fMaxStagBest =int( fCurNumOfGen / 10 );                  /* fMaxStagBest = G/10 (See Section 2.2) */
    } 
    else if( fMaxStagBest != 0 && fMaxStagBest <= fStagBest ){ /* Terminate Stage I (proceed to Stage II) */
      fStagBest = 0;
      fMaxStagBest = 0;
      fCurNumOfGen1 = fCurNumOfGen;
      fFlagC[ 1 ] = 2; 
      fStage = 2;      
    }
    return false;
  }

  if( fStage == 2 ){ /* Stage II */
    if( fStagBest == int(1500/fNumOfKids) && fMaxStagBest == 0 ){ /* 1500/N_ch (See Section 2.2) */
      fMaxStagBest = int( (fCurNumOfGen - fCurNumOfGen1) / 10 ); /* fMaxStagBest = G/10 (See Section 2.2) */
    } 
    else if( fMaxStagBest != 0 && fMaxStagBest <= fStagBest ){ /* Terminate Stage II and GA */
      return true;
    }

    return false;
  }

  return false;
}


void TEnvironment::SetAverageBest() 
{
  Type_dis stockBest = tBest.fEvaluationValue;
  
  fAverageValue = 0.0;
  fBestIndex = 0;
  fBestValue = tCurPop[0].fEvaluationValue;
  
  for(int i = 0; i < fNumOfPop; ++i ){
    fAverageValue += tCurPop[i].fEvaluationValue;
    if( tCurPop[i].fEvaluationValue < fBestValue ){
      fBestIndex = i;
      fBestValue = tCurPop[i].fEvaluationValue;
    }
  }
  
  tBest = tCurPop[ fBestIndex ];
  fAverageValue /= (Type_dou)fNumOfPop;

  if( tBest.fEvaluationValue < stockBest ){
    fStagBest = 0;
    fBestNumOfGen = fCurNumOfGen;
    fBestAccumeratedNumCh = fAccumurateNumCh;
  }
  else ++fStagBest;
}


void TEnvironment::InitPop()
{



  for ( int i = 0; i < fNumOfPop; ++i ){ 


    tKopt->MakeRandSol( tCurPop[ i ] );    /* Make a random tour */
    tKopt->DoIt( tCurPop[ i ] );           /* Apply the local search with the 2-opt neighborhood */
    fEvaluator->TranceLinkOrder( tCurPop[ i ] );  // Large

    if(i==0){
    	tBest = tCurPop[ i ];
    }
    else if(tBest.fEvaluationValue>tCurPop[i].fEvaluationValue){
    	tBest = tCurPop[ i ];
    }

	if(((clock() - this->fTimeStart)/(double)CLOCKS_PER_SEC)>fEvaluator->Ncity/5.0) {
		std::cout<<"not finish the initial population"<<std::endl;
		return;
	}

  }
}


void TEnvironment::SelectForMating()
{
  /* fIndexForMating[] <-- a random permutation of 0, ..., fNumOfPop-1 */
  tRand->Permutation( fIndexForMating, fNumOfPop, fNumOfPop ); 
  fIndexForMating[ fNumOfPop ] = fIndexForMating[ 0 ];
}

void TEnvironment::SelectForSurvival( int s )
{
}


void TEnvironment::GenerateKids( int s )
{
  tCross->SetParents( tCurPop[fIndexForMating[s]], tCurPop[fIndexForMating[s+1]], fFlagC, fNumOfKids );  
  
  /* Note: tCurPop[fIndexForMating[s]] is replaced with a best offspring solutions in tCorss->DoIt(). 
     fEegeFreq[][] is also updated there. */
  tCross->DoIt( tCurPop[fIndexForMating[s]], tCurPop[fIndexForMating[s+1]], fNumOfKids, 1, fFlagC, fEdgeFreq );

  fAccumurateNumCh += tCross->fNumOfGeneratedCh;
}


void TEnvironment::GetEdgeFreq()  // Large
{
  int N = fEvaluator->Ncity;
  int k0, k1;
  
  for( int j1 = 0; j1 < N; ++j1 )
    for( int j2 = 0; j2 < fEvaluator->fNearNumMax; ++j2 ) 
      fEdgeFreq[ j1 ][ j2 ] = 0;

  
  for( int i = 0; i < fNumOfPop; ++i )
  {
    for(int j = 0; j < N; ++j )
    {
      k0 = tCurPop[ i ].fOrder[ j ][ 0 ];      
      k1 = tCurPop[ i ].fOrder[ j ][ 1 ];
      if( k0 != -1 ) 
	++fEdgeFreq[ j ][ k0 ];
      if( k1 != -1 ) 
      ++fEdgeFreq[ j ][ k1 ];
    }
  }
}


void TEnvironment::PrintOn( int n, char* dstFile ) 
{
  printf( "n = %d val = %d Gen = %d Time = %d %d\n" ,
	  n,
	  tBest.fEvaluationValue,
	  fCurNumOfGen,
	  (int)((double)(this->fTimeInit - this->fTimeStart)/(double)CLOCKS_PER_SEC),
	  (int)((double)(this->fTimeEnd - this->fTimeStart)/(double)CLOCKS_PER_SEC) );
  fflush(stdout);

  FILE *fp;
  char filename[ 80 ];
  sprintf( filename, "%s_Result", dstFile );
  fp = fopen( filename, "a");

  fprintf( fp, "%d %d %d %d %d\n" ,
	   n,
	   tBest.fEvaluationValue,
	   fCurNumOfGen,
	   (int)((double)(this->fTimeInit - this->fTimeStart)/(double)CLOCKS_PER_SEC),
	   (int)((double)(this->fTimeEnd - this->fTimeStart)/(double)CLOCKS_PER_SEC) );

  fclose( fp );
}


void TEnvironment::WriteBest( char* dstFile ,std::string fileS)
{
  FILE *fp;
  char filename[ 80 ];
  sprintf( filename, "%s_BestSol", dstFile );
  fp = fopen( filename, "a");
  
  fEvaluator->WriteTo( fp, tBest );

  fclose( fp );

	std::ofstream outf(fileS);
	if(outf.is_open()){
		outf.setf(std::ios::fixed);
		outf<<"1 "<<tBest.fEvaluationValue<<" "<<((double)(fEvaluator->fTInit - fEvaluator->fTStart)/(double)CLOCKS_PER_SEC)<<" "\
				<<((double)(clock() - this->fTimeStart)/(double)CLOCKS_PER_SEC)\
				<<std::endl;
		outf.close();
	}

}

void TEnvironment::FetchSol(std::vector<int> &sol, long int &finalv){
	fEvaluator->transSol( sol, tBest );
	fEvaluator->newdoIt(tBest, finalv);
}


void TEnvironment::WritePop( int n, char* dstFile ) 
{
  FILE *fp;
  char filename[ 80 ];
  sprintf( filename, "%s_POP_%d", dstFile, n );
  fp = fopen( filename, "w");

  for( int s = 0; s < fNumOfPop; ++s )
    fEvaluator->WriteTo( fp, tCurPop[ s ] );

  fclose( fp );
}


void TEnvironment::ReadPop( char* fileName )
{
  FILE* fp;

  if( ( fp = fopen( fileName, "r" ) ) == NULL ){
    printf( "Read Error1\n"); 
    fflush( stdout );
    exit( 1 );
  }

  for ( int i = 0; i < fNumOfPop; ++i ){ 
    if( fEvaluator->ReadFrom( fp, tCurPop[ i ] ) == false ){
      printf( "Read Error2\n"); 
      fflush( stdout );
      exit( 1 );
    }
    fEvaluator->TranceLinkOrder( tCurPop[ i ] );  // Large
  }
  fclose( fp );
}


