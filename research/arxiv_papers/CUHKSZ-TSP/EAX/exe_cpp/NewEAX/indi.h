#ifndef __INDI__
#define __INDI__


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef long long int Type_dis ;
typedef long double Type_dou ;


class TIndi {
public:
  TIndi();
  ~TIndi();
  void Define( int N );
  TIndi& operator = ( const TIndi& src );
  bool operator == (  const TIndi& indi2 );

  int fN;
  int** fLink;
  int** fOrder;         // Large
  Type_dis fEvaluationValue;
};


#endif

