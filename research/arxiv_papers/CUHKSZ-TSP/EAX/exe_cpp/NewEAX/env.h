#ifndef __ENVIRONMENT__
#define __ENVIRONMENT__

#ifndef __INDI__
#include "indi.h"
#endif

#ifndef __RAND__
#include "rand.h"
#endif

#ifndef __EVALUATOR__
#include "evaluator.h"
#endif

#ifndef __Cross__
#include "cross.h"
#endif

#ifndef __KOPT__
#include "kopt.h"
#endif

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <fstream>

class TEnvironment {
public:
  TEnvironment(); 
  ~TEnvironment();
  bool Define();                         /* Define the variables */
  bool ThirdDefine(int Sub_TSP_City_Num, const std::vector<double> &x, const std::vector<double> &y,
			const std::vector<std::vector<int>> &adjedge);

  void DoIt();                           /* Main procedure of the GA */
  void Init();                           /* Initialization of the GA */
  bool TerminationCondition();           /* Decide whether to proceed to next stage 
					    (or treminate the GA) */
  void SetAverageBest();                 /* Compute average and best tour lengths of the population */
  void InitPop();                        /* Create an initial population */
  void SelectForMating();                /* Determine a set of pairs of parents at each generation */
  void SelectForSurvival( int s );       /* Not do anything */
  void GenerateKids( int s );            /* Generate offspring solutions from a selected pair of  
                                            parents. Selection for survival is also performed here. */
  void GetEdgeFreq();                    /* Compute the frequency of the edges of the population */

  void PrintOn( int n, char* dstFile );  /* Display and write summary of results */
  void WriteBest( char* dstFile, std::string fileS );       /* Write the best tour */
  void WritePop( int n, char* dstFile ); /* Write the population */
  void ReadPop( char* fileName );        /* Read an initial population */

  void FetchSol(std::vector<int> &sol, long int &finalv);


  TEvaluator* fEvaluator;                /* Distance of the edges */
  TCross* tCross;                        /* Edge assembly crossover */
  TKopt* tKopt;                          /* Local search with the 2-opt neighborhood */
  char *fFileNameTSP;                    /* File name of an TSP instance */
  char *fFileNameInitPop;                /* File name of an initial population */

  int fNumOfPop;                         /* Number of population members (N_pop in the paper) */
  int fNumOfKids;                        /* Number of offspring solutions (N_ch in the paper) */
  TIndi* tCurPop;                        /* Current population members */
  TIndi tBest;                           /* Best solution in the current population */
  int fCurNumOfGen;                      /* The current number of generations */
  long int fAccumurateNumCh;             /* The accumulated number of offspring solutions */
  int fBestNumOfGen;                     /* The number of generations at which the current best 
                                            solution was found */ 
  long int fBestAccumeratedNumCh;        /* The accumulated number of offspring solutions at which 
                                            the current best solution was found */
  int **fEdgeFreq;                       /* The frequency of the edges of the population */
  Type_dou fAverageValue;                  /* The average tour lengths of the population */
  Type_dis fBestValue;                        /* The tour lenght of the best tour in the population */
  int fBestIndex;                        /* Index of the best population member */
  int* fIndexForMating;                  /* Mating list (r[] in the paper) */

  int fStagBest;                         /* The number of generations during which no improvement  
                                            is found in the best tour */
  int fFlagC[ 10 ];                      /* Specify configurations of EAX and selection strategy */
  int fStage;                            /* Current stage */
  int fMaxStagBest;                      /* If fStagBest = fMaxStagBest, proceed to the next stage */
  int fCurNumOfGen1;                     /* Number of generations at which Stage I is terminated */

  clock_t fTimeStart, fTimeInit, fTimeEnd;  /* Use them to measure the execution time */
  double PreprocessingTime;
  double RunningTime;
  double AllTime;

};


#endif

