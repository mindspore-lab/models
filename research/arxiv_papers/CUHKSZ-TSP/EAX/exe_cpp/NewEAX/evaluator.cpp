//#ifndef __EVALUATOR__
#include "evaluator.h"
//#endif

TEvaluator::TEvaluator()
{
  fEdgeDisOrder = NULL;
  fNearCity = NULL;
  Ncity = 0;
  fNearNumMax = 100;                               // Large

  replaceDis=-2000;
  fixPlus=0;

  x=NULL;
  y=NULL;

  fTStart=clock();
  fTInit=clock();
  fTEnd=clock();
}

TEvaluator::~TEvaluator()
{
  for ( int i = 0; i < Ncity; ++i ) 
    delete [] fEdgeDisOrder[ i ];
  delete [] fEdgeDisOrder;
  for ( int i = 0; i < Ncity; ++i ) 
    delete [] fNearCity[ i ];
  delete [] fNearCity;
//  for ( int i = 0; i < Ncity; ++i )
//    delete [] fEdgeDisOrder[ i ];                   // Large
//  delete [] fEdgeDisOrder;
  
  delete [] x;
  delete [] y;
}

bool TEvaluator::SetInstance( char filename[] )
{
  FILE* fp;
  int flag;
  int n;
  char word[ 80 ];
  Type_dis *DisTmp;

  fp = fopen( filename, "r" );

  ////// read instance //////
  while( 1 ){
    if( fscanf( fp, "%s", word ) == EOF )
      break;

    if( strcmp( word, "DIMENSION" ) == 0 ){
      fscanf( fp, "%s", word ); 
      assert( strcmp( word, ":" ) == 0 );
      fscanf( fp, "%d", &Ncity ); 
    } 

    if( strcmp( word, "EDGE_WEIGHT_TYPE" ) == 0 ){
      fscanf( fp, "%s", word ); 
      assert( strcmp( word, ":" ) == 0 );
      fscanf( fp, "%s", fType ); 
    } 

    if( strcmp( word, "NODE_COORD_SECTION" ) == 0 ) 
      break;
      

  }
  if( strcmp( word, "NODE_COORD_SECTION" ) != 0 ){
    printf( "Error in reading the instance\n" );
    exit(0);
  }

  x = new double [ Ncity ]; 
  y = new double [ Ncity ]; 
  int checkedN[ Ncity ];

  int xi, yi; 
  for( int i = 0; i < Ncity; ++i ) 
  {
    fscanf( fp, "%d", &n );
    assert( i+1 == n ); 
    fscanf( fp, "%s", word ); 
    x[ i ] = atof( word );
    fscanf( fp, "%s", word ); 
    y[ i ] = atof( word );
  }

  fclose(fp);
  //////////////////////////

  fEdgeDisOrder = new Type_dis* [ Ncity ];         // Large
  for( int i = 0; i < Ncity; ++i ) 
    fEdgeDisOrder[ i ] = new Type_dis [ fNearNumMax+1 ];
  fNearCity = new int* [ Ncity ];
  for( int i = 0; i < Ncity; ++i ) 
    fNearCity[ i ] = new int [ fNearNumMax+1 ];

  DisTmp = new Type_dis [ Ncity ];                  // Large
  int city_num;
  Type_dis min_dis;

  if( strcmp( fType, "EUC_2D" ) != 0 ){        
    printf( "Please modify program code in TEvaluator::Direct( ) appropriately when EDGE_WEIGHT_TYPE is not EUC_2D.\n" );
    exit( 1 );  /* After the modification, remove this line. */
  }


  /***************************************************************/
	fTStart=clock();
	bool flag_T=true;
  /***************************************************************/

  for( int i = 0; i < Ncity ; ++i )
  {
    if( strcmp( fType, "EUC_2D" ) == 0  ) {
      for( int j = 0; j < Ncity ; ++j )
	DisTmp[ j ] = (Type_dis)(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))+0.5);
    }
    else if( strcmp( fType, "ATT" ) == 0  ) { 
      for( int j = 0; j < Ncity ; ++j ){
	double r = (sqrt(((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))/10.0));
	int t = (int)r;
	if( (double)t < r ) DisTmp[ j ] = t+1;
	else DisTmp[ j ] = t; 
      }
    }
    else if( strcmp( fType, "CEIL_2D" ) == 0  ) {  
      for( int j = 0; j < Ncity ; ++j )
	DisTmp[ j ] = (Type_dis)ceil(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])));
    }
    else{
      printf( "EDGE_WEIGHT_TYPE is not supported\n" );
      exit( 1 );
    }


//    /*************************************/
    fTInit=clock();
    if(((double)(this->fTInit - this->fTStart)/(double)CLOCKS_PER_SEC)>Ncity/5.0) {
    	std::cout<<"IniTIME: "<< (double)(this->fTInit - this->fTStart)/(double)CLOCKS_PER_SEC<<std::endl;
    	delete [] DisTmp;
    	return false;
    }
//    /*************************************/


    for( int j3 = 0; j3 < Ncity; ++j3 )         // Large
      checkedN[ j3 ] = 0;

    checkedN[ i ] = 1;                          // Large
    fNearCity[ i ][ 0 ] = i;                    // Large
    fEdgeDisOrder[ i ][ 0 ] = 0;                // Large
    
    for( int j1 = 1; j1 <= fNearNumMax; ++j1 )  // Large
    {
      min_dis = 100000000;
      for( int j2 = 0; j2 < Ncity ; ++j2 )
      { 
	if( DisTmp[ j2 ] <= min_dis && checkedN[ j2 ] == 0 )
	{
	  city_num = j2;
	  min_dis = DisTmp[ j2 ];
	}
      }
      fNearCity[ i ][ j1 ] = city_num;
      fEdgeDisOrder[ i ][ j1 ] = min_dis;
      checkedN[ city_num ] = 1;
    }
  }

  delete [] DisTmp;

  fTInit=clock();
	std::cout<<"PreprocessingTIME: "<< (double)(this->fTInit - this->fTStart)/(double)CLOCKS_PER_SEC<<std::endl;
	return true;
}


bool TEvaluator::ThirdSet(int Sub_TSP_City_Num, const std::vector<double> &cx, const std::vector<double> &cy,
			const std::vector<std::vector<int>> &adjedge){
	Ncity=Sub_TSP_City_Num;
	x = new double [ Ncity ];
	y = new double [ Ncity ];
	int checkedN[ Ncity ];
	Type_dis *DisTmp;
	DisTmp = new Type_dis [ Ncity ];
	//////////////////////////
	ADJEDGE=std::vector<std::vector<int>> (Ncity);
	for(int i=0;i<Ncity;++i) {
		x[i]=cx[i];
		y[i]=cy[i];
		for(unsigned int j=0; j<adjedge[i].size(); ++j) ADJEDGE[i].emplace_back(adjedge[i][j]);
	}
	std::cout<<"adjedge is loaded"<<std::endl;

	/*****************************************/

	fTStart=clock();
	bool flag_T=true;
	/*****************************************/

	fEdgeDisOrder = new Type_dis* [ Ncity ];         // Large
	for( int i = 0; i < Ncity; ++i )	fEdgeDisOrder[ i ] = new Type_dis [ fNearNumMax+1 ];
	fNearCity = new int* [ Ncity ];
	for( int i = 0; i < Ncity; ++i )	fNearCity[ i ] = new int [ fNearNumMax+1 ];

	                 // Large
	int city_num;
	Type_dis dds=0;
	Type_dis min_dis;
	Type_dis ct=0;

	for( int i = 0; i < Ncity ; ++i ){
	    for( int j3 = 0; j3 < Ncity; ++j3 )         // Large
	      checkedN[ j3 ] = 0;
	    for(int k=0;k<Ncity;++k) {
	    	DisTmp[k]=Direct(i,k);
	    	if(dds<DisTmp[k]) dds=DisTmp[k];			// find the maximum dis
	    	if(DisTmp[k]<0) ct+=1;
	    }

	    checkedN[ i ] = 1;                          // Large
	    fNearCity[ i ][ 0 ] = i;                    // Large
	    fEdgeDisOrder[ i ][ 0 ] = 0;                // Large


	    /*************************************/
	    fTInit=clock();
	    if(((double)(this->fTInit - this->fTStart)/(double)CLOCKS_PER_SEC)>Ncity*1.0) {
	    	std::cout<<"IniTIME: "<< (double)(this->fTInit - this->fTStart)/(double)CLOCKS_PER_SEC<<std::endl;
	    	delete [] DisTmp;
	    	return false;
	    }

	    /*************************************/

	    for( int j1 = 1; j1 <= fNearNumMax; ++j1 ){  // Large

			min_dis = dds+100000000;
			for( int j2 = 0; j2 < Ncity ; ++j2 ){
				if( DisTmp[ j2 ] <= min_dis && checkedN[ j2 ] == 0 ){
					city_num = j2;
					min_dis = DisTmp[ j2 ];
				}
			}
			fNearCity[ i ][ j1 ] = city_num;
			fEdgeDisOrder[ i ][ j1 ] = min_dis;
			checkedN[ city_num ] = 1;
	    }

	}

	std::cout<<"NearCity is loaded"<<std::endl;

	replaceDis=-dds*2;
	for(int i=0;i<Ncity;++i){

		   /*************************************/
	    fTInit=clock();
	    if(((double)(this->fTInit - this->fTStart)/(double)CLOCKS_PER_SEC)>Ncity*1.0) {
	    	std::cout<<"IniTIME: "<< (double)(this->fTInit - this->fTStart)/(double)CLOCKS_PER_SEC<<std::endl;
	    	delete [] DisTmp;
	    	return false;
	    }
	    /*************************************/

		for(int j=1;j<=fNearNumMax;++j){
			if(fEdgeDisOrder[i][j]<0) fEdgeDisOrder[i][j]=replaceDis;
			else break;
		}
	}

	std::cout<<"replaceDis is replaced"<<std::endl;
//	std::cout<<"replaceDis="<<replaceDis<<std::endl;
//
//	for(int i=0;i<Ncity;++i){
//		std::cout<<i<<" ";
//		for(int j=1;j<=fNearNumMax;++j){
//			if(fEdgeDisOrder[i][j]<0) std::cout<<fEdgeDisOrder[i][j]<<std::endl;
//			else break;
//		}
//
//	}
//	exit(0);

	fixPlus=-ct/2*replaceDis;


	delete [] DisTmp;

    fTInit=clock();
	std::cout<<"IniTIME: "<< (double)(this->fTInit - this->fTStart)/(double)CLOCKS_PER_SEC<<std::endl;
	return true;
}



void TEvaluator::DoIt( TIndi& indi )
{
  Type_dis d;
  d = 0;  
  for(int i = 0; i < Ncity; ++i )
  {  
    d = d + this->Direct( i, indi.fLink[i][0] );
    d = d + this->Direct( i, indi.fLink[i][1] );
  }
  indi.fEvaluationValue = d/2;
}


Type_dis TEvaluator::Direct( int i, int j )  // Large
{
//  int d, t;
//  double r;
//
//
//  /* If EUC_2D is not used, plese write an distance function you want to use.
//     This is because the use of ''if statement'' described above is time-consuming. */
//
  // EUC_2D:

	Type_dis d;

	d = (Type_dis)(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))+0.5);

	return d;

//	if(checkexist(n1, n2)) return replaceDis;
//
//	double lati, latj, longi, longj;
//	double q1, q2, q3, q4, q5;
//
//	lati = x[n1] * PI / 180.0;
//	latj = x[n2] * PI / 180.0;
//
//	longi = y[n1] * PI / 180.0;
//	longj = y[n2] * PI / 180.0;
//
//	q1 = cos(latj) * sin(longi - longj);
//	q3 = sin((longi - longj) / 2.0);
//	q4 = cos((longi - longj) / 2.0);
//	q2 = sin(lati + latj) * q3 * q3 - sin(lati - latj) * q4 * q4;
//	q5 = cos(lati - latj) * q4 * q4 - cos(lati + latj) * q3 * q3;
//	return (Type_dis)(6378388.0 * atan2(sqrt(q1 * q1 + q2 * q2), q5) + 1.0);


}

bool TEvaluator::checkexist(int base, int n){
	for(unsigned int i=0;i<ADJEDGE[base].size();++i)
		if(n==ADJEDGE[base][i]) return true;
	return false;
}


void TEvaluator::TranceLinkOrder( TIndi& indi )  // Large
{
  int a, b;
  for(int i = 0; i < Ncity; ++i )
  {  
    a = i;
    b = indi.fLink[ i ][ 0 ];
    indi.fOrder[ i ][ 0 ] = this->GetOrd( a, b );

    b = indi.fLink[ i ][ 1 ];
    indi.fOrder[ i ][ 1 ] = this->GetOrd( a, b );
  }
}


int TEvaluator::GetOrd( int a, int b )  // Large
{
  for( int s = 0; s <= fNearNumMax; ++s )
  {
    if( fNearCity[ a ][ s ] == b )
      return s;
  }
  return -1;
}


void TEvaluator::WriteTo( FILE* fp, TIndi& indi ) 
{
  assert( Ncity == indi.fN );
  int Array[ Ncity ];
  int curr, next, pre, st, count;

  count = 0;
  pre = -1;
  curr = 0;
  st = 0;
  while( 1 )
  {
    Array[ count++ ] = curr + 1;
//    Array[ count++ ] = curr;

    if( count > Ncity ){
      printf( "Invalid\n" );
      return;
    } 
 
    if( indi.fLink[ curr ][ 0 ] == pre )
      next = indi.fLink[ curr ][ 1 ];
    else 
      next = indi.fLink[ curr ][ 0 ];

    pre = curr;
    curr = next;
    if( curr == st )
      break;
  }

  if( this->CheckValid( Array, indi.fEvaluationValue ) == false ){
    printf( "Individual is invalid \n" );
  }

  fprintf( fp, "%d %d\n", indi.fN, indi.fEvaluationValue );
  for( int i = 0; i < indi.fN; ++i )
    fprintf( fp, "%d ", Array[ i ] -1 );
  fprintf( fp, "\n" ); 
}

void TEvaluator::transSol(std::vector<int> &sol ,TIndi& indi ){
	  assert( Ncity == indi.fN );
	  int Array[ Ncity ];
	  int curr, next, pre, st, count;

	  count = 0;
	  pre = -1;
	  curr = 0;
	  st = 0;
	  while( 1 )
	  {
	    Array[ count++ ] = curr;

	    if( count > Ncity ){
	      printf( "Invalid\n" );
	      return;
	    }

	    if( indi.fLink[ curr ][ 0 ] == pre )
	      next = indi.fLink[ curr ][ 1 ];
	    else
	      next = indi.fLink[ curr ][ 0 ];

	    pre = curr;
	    curr = next;
	    if( curr == st )
	      break;
	  }

	  for(int i=0;i<Ncity;++i) sol[i]=Array[i];
}

void TEvaluator::newdoIt(TIndi& indi, long int &finalv){

	  finalv=0;
	  long long int tmpd=0;
	  for(int i = 0; i < Ncity; ++i )
	  {
		  tmpd=this->Direct( i, indi.fLink[i][0] );
		  if (tmpd>0) finalv += tmpd;
		  tmpd=this->Direct( i, indi.fLink[i][1] );
		  if (tmpd>0) finalv += tmpd;
	  }
	  finalv /=2;
}


bool TEvaluator::ReadFrom( FILE* fp, TIndi& indi )
{
  assert( Ncity == indi.fN );
  int Array[ Ncity ];
  int curr, next, pre, st, count;
  int N, value;

  if( fscanf( fp, "%d %d", &N, &value ) == EOF ) 
    return false;
  assert( N == Ncity );
  indi.fN = N;
  indi.fEvaluationValue = value;

  for( int i = 0; i < Ncity; ++i ){ 
    if( fscanf( fp, "%d", &Array[ i ] ) == EOF )
      return false;
  }

  if( this->CheckValid( Array, indi.fEvaluationValue ) == false ){
    printf( "Individual is invalid \n" );
    return false;
  }

  for( int i = 0; i < Ncity; ++i ){ 
    Array[ i ] -= 1; 
  }

  for( int i = 1; i < Ncity-1; ++i ){ 
    indi.fLink[ Array[ i ] ][ 0 ] = Array[ i-1 ]; 
    indi.fLink[ Array[ i ] ][ 1 ] = Array[ i+1 ]; 
  }
  indi.fLink[ Array[ 0 ] ][ 0 ] = Array[ Ncity-1 ]; 
  indi.fLink[ Array[ 0 ] ][ 1 ] = Array[ 1 ]; 
  indi.fLink[ Array[ Ncity-1 ] ][ 0 ] = Array[ Ncity-2 ]; 
  indi.fLink[ Array[ Ncity-1 ] ][ 1 ] = Array[ 0 ]; 

  return true;
}    


bool TEvaluator::CheckValid( int* array, int value )
{
  int check[ Ncity ];
  int distance;

  for( int i = 0; i < Ncity; ++i ){
    check[ i ] = 0;
  }

  for( int i = 0; i < Ncity; ++i )
    ++check[ array[ i ]-1 ];

  for( int i = 0; i < Ncity; ++i ){
    if( check[ i ] != 1 ){
      return false;
    }
  }
    
  distance = 0;  
  for( int i = 0; i < Ncity-1; ++i ){
    distance += this->Direct( array[ i ]-1, array[ i+1 ]-1 );   // Large
  }
  distance += this->Direct( array[ Ncity-1 ]-1, array[ 0 ]-1 ); // Large
  if( distance != value ){
    return false;
  }
  return true;
}
