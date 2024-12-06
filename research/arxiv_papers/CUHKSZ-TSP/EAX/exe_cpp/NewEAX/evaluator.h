#ifndef __EVALUATOR__
#define __EVALUATOR__

//#ifndef __INDI__
#include "indi.h"
//#endif


#include <string.h>
#include <assert.h>
#include <stdio.h>

#include <vector>
#include <iostream>

#define PI 3.14159265358979323846264

class TEvaluator {
 public:
  TEvaluator();
  ~TEvaluator();
  bool SetInstance( char filename[] );
  bool ThirdSet(int Sub_TSP_City_Num, const std::vector<double> &x, const std::vector<double> &y,
			const std::vector<std::vector<int>> &adjedge);
  void DoIt( TIndi& indi );
  Type_dis Direct( int i, int j );           // Large
  bool checkexist(int n1, int n2);


  void TranceLinkOrder( TIndi& indi );  // Large
  int GetOrd( int a, int b );           // Large

  void WriteTo( FILE* fp, TIndi& indi );
  bool ReadFrom( FILE* fp, TIndi& indi );   
  bool CheckValid( int* array, int value ); 

  void transSol(std::vector<int> &sol, TIndi& indi );
  void newdoIt(TIndi& indi, long int &finalv);

  char fType[ 80 ];                      // Large 
  int fNearNumMax;
  int **fNearCity;
  Type_dis **fEdgeDisOrder;                   // Large
  int Ncity;
  double *x;
  double *y;

  std::vector<std::vector<int>> ADJEDGE;

  Type_dis replaceDis;
  Type_dis fixPlus;


  clock_t fTStart, fTInit, fTEnd;  /* Use them to measure the execution time */

};
  

#endif



