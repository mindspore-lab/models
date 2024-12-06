/*
 * evaluator.cpp
 *   created on: April 24, 2013
 * last updated: May 10, 2020
 *       author: Shujia Liu
 */

#ifndef __EVALUATOR__
#include "evaluator.h"
#endif
#include <math.h>
#include <iostream>
using namespace std;

TEvaluator::TEvaluator() {
	Ncity = 0;
	fNearNumMax = 50;
}

TEvaluator::~TEvaluator() {}



void TEvaluator::setInstance(const string& filename) {
	FILE* fp;
	int n __attribute__((unused));
	char word[ 80 ], type[ 80 ];
	fp = fopen(filename.c_str(), "r");
	while( 1 ){
		if( fscanf( fp, "%s", word ) == EOF ) break;
//		if( strcmp( word, "DIMENSION" ) == 0 ){
//			fscanf( fp, "%s", word );
//			fscanf( fp, "%d", &Ncity );
//		}
//		if( strcmp( word, "EDGE_WEIGHT_TYPE" ) == 0 ){
//			fscanf( fp, "%s", word );
//			fscanf( fp, "%s", type );
//		}
		if( strcmp( word, "NODE_COORD_SECTION" ) == 0 ) break;
	}
	if( strcmp( word, "NODE_COORD_SECTION" ) != 0 ){
		printf( "Error in reading the instance\n" );
		exit(0);
	}
	x.resize(Ncity);
	y.resize(Ncity);
	vector<int> checkedN(Ncity);

//	for( int i = 0; i < Ncity; ++i ){
//		fscanf( fp, "%d", &n );
//		fscanf( fp, "%s", word );
//		x[ i ] = atof( word );
//		fscanf( fp, "%s", word );
//		y[ i ] = atof( word );
//	}
	fclose(fp);

	fEdgeDis.clear();
	for (int i = 0; i < Ncity; i++) {
		vector<Type_dis> row(Ncity);
		fEdgeDis.push_back(row);
	}

	fNearCity.clear();
	for (int i = 0; i < Ncity; i++) {
		vector<int> row(fNearNumMax + 1);
		fNearCity.push_back(row);
	}

	if( strcmp( type, "EUC_2D" ) == 0  ) {
		std::cout << fEdgeDis.size() << " " << fEdgeDis[0].size() << std::endl;
		for( int i = 0; i < Ncity ; ++i )
			for( int j = 0; j < Ncity ; ++j )
				fEdgeDis[ i ][ j ]=(int)(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))+0.5);
	} else if ( strcmp( type, "ATT" ) == 0  ) {
		for( int i = 0; i < Ncity; ++i ){
			for( int j = 0; j < Ncity; ++j ) {
				double r = (sqrt(((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))/10.0));
				int t = (int)r;
				if( (double)t < r ) {
					fEdgeDis[ i ][ j ] = t+1;
				} else {
					fEdgeDis[ i ][ j ] = t;
				}
			}
		}
	} else if (strcmp( type, "CEIL_2D" ) == 0) {
		for (int i = 0; i < Ncity ; ++i) {
			for (int j = 0; j < Ncity ; ++j) {
				fEdgeDis[ i ][ j ]=(int)ceil(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])));
			}
		}
	} else {
		printf( "EDGE_WEIGHT_TYPE is not supported\n" );
		exit( 1 );
	}
	int ci, j1, j2, j3;
	int cityNum = 0;
	int minDis;
	for( ci = 0; ci < Ncity; ++ci ){
		for( j3 = 0; j3 < Ncity; ++j3 ) checkedN[ j3 ] = 0;
		checkedN[ ci ] = 1;
		fNearCity[ ci ][ 0 ] = ci;
		for( j1 = 1; j1 <= fNearNumMax; ++j1 ) {
			minDis = 100000000;
			for( j2 = 0; j2 < Ncity; ++j2 ){
				if( fEdgeDis[ ci ][ j2 ] <= minDis && checkedN[ j2 ] == 0 ){
					cityNum = j2;
					minDis = fEdgeDis[ ci ][ j2 ];
				}
			}
			fNearCity[ ci ][ j1 ] = cityNum;
			checkedN[ cityNum ] = 1;
		}
	}
}

void TEvaluator::setMatrix(int Sub_TSP_City_Num, int **Temp_Distance){

	Ncity=Sub_TSP_City_Num;

	x.resize(Ncity);
	y.resize(Ncity);
	vector<int> checkedN(Ncity);


	fEdgeDis.clear();
	for (int i = 0; i < Ncity; i++) {
		vector<Type_dis> row(Ncity,0);
		fEdgeDis.push_back(row);
	}

	fNearCity.clear();
	for (int i = 0; i < Ncity; i++) {
		vector<int> row(fNearNumMax + 1);
		fNearCity.push_back(row);
	}

	Type_dis maxi=0;
	Type_dis replace_dis=0;
	Type_dis tmpdis=-1;
	Type_dis ct=0;

	for(int i=0;i<Ncity;++i){
		for(int j=i+1;j<Ncity;++j){
			if(Temp_Distance[i][j]>maxi) maxi=Temp_Distance[i][j];
		}
	}
//	replace_dis=-maxi*Ncity/100;
	replace_dis=-maxi*2;

//	std::cout<<ct<<std::endl;
//	std::cout<<replace_dis<<std::endl;


	for(int i=0;i<Ncity;++i){
		for(int j=i+1;j<Ncity;++j){
			if(Temp_Distance[i][j]<0) {
				tmpdis=replace_dis;
				ct+=1;
			}
			else {
				tmpdis=Temp_Distance[i][j];
			}

			fEdgeDis[i][j]=tmpdis;
			fEdgeDis[j][i]=tmpdis;
		}
	}
	fixcplus=-replace_dis*ct;

	/***************************************************************************************/
	int ci, j1, j2, j3;
	int cityNum = 0;
	int minDis;
	for( ci = 0; ci < Ncity; ++ci ){
		for( j3 = 0; j3 < Ncity; ++j3 ) checkedN[ j3 ] = 0;
		checkedN[ ci ] = 1;
		fNearCity[ ci ][ 0 ] = ci;
		for( j1 = 1; j1 <= fNearNumMax; ++j1 ) {
			minDis = 100000000;
			for( j2 = 0; j2 < Ncity; ++j2 ){
				if( fEdgeDis[ ci ][ j2 ] <= minDis && checkedN[ j2 ] == 0 ){
					cityNum = j2;
					minDis = fEdgeDis[ ci ][ j2 ];
				}
			}
			fNearCity[ ci ][ j1 ] = cityNum;
			checkedN[ cityNum ] = 1;
		}
	}
}

void TEvaluator::doIt( TIndi& indi ) {
	Type_dis d = 0;
	for( int i = 0; i < Ncity; ++i ) d += fEdgeDis[ i ][ indi.fLink[i][0] ] + fEdgeDis[ i ][ indi.fLink[i][1] ];
	indi.fEvaluationValue = d/2;
}

Type_dis TEvaluator::AnodoIt( TIndi& indi, int **Mdis ) {
	Type_dis d = 0;
	for( int i = 0; i < Ncity; ++i ) {
		if(Mdis[ i ][ indi.fLink[i][0] ]>0) d+=Mdis[ i ][ indi.fLink[i][0] ];
		else d-=Mdis[ i ][ indi.fLink[i][0] ];

		if(Mdis[ i ][ indi.fLink[i][1] ]>0) d+=Mdis[ i ][ indi.fLink[i][1] ];
		else d-=Mdis[ i ][ indi.fLink[i][1] ];
//		d += fEdgeDis[ i ][ indi.fLink[i][0] ] + fEdgeDis[ i ][ indi.fLink[i][1] ];
	}

	return d/2;
//	indi.fEvaluationValue = d/2;
}

Type_dis TEvaluator::SecdiIt(TIndi& indi, int **Mdis){
	Type_dis d = 0;
	for( int i = 0; i < Ncity; ++i ) {
		if(Mdis[ i ][ indi.fLink[i][0] ]>0) d+=Mdis[ i ][ indi.fLink[i][0] ];
//		else d-=Mdis[ i ][ indi.fLink[i][0] ];

		if(Mdis[ i ][ indi.fLink[i][1] ]>0) d+=Mdis[ i ][ indi.fLink[i][1] ];
//		else d-=Mdis[ i ][ indi.fLink[i][1] ];
//		d += fEdgeDis[ i ][ indi.fLink[i][0] ] + fEdgeDis[ i ][ indi.fLink[i][1] ];
	}

	return d/2;
}

void TEvaluator::writeTo( FILE* fp, TIndi& indi ){
	Array.resize(Ncity);
	int curr=0, st=0, count=0, pre=-1, next;
	while( 1 ){
//		Array[ count++ ] = curr + 1;
		Array[ count] = curr;
		count++;
		if( count > Ncity ){
			printf( "Invalid\n" );
			return;
		}
		if( indi.fLink[ curr ][ 0 ] == pre ) next = indi.fLink[ curr ][ 1 ];
		else next = indi.fLink[ curr ][ 0 ];

		pre = curr;
		curr = next;
		if( curr == st ) break;
	}
//	if( this->checkValid( Array, indi.fEvaluationValue ) == false )	printf( "Individual is invalid \n" );

	fprintf( fp, "%d\n\n", indi.fN);
	for( int i = 0; i < indi.fN; ++i )
		fprintf( fp, "%d\n", Array[ i ] );
//	fprintf( fp, "\n" );
}

void TEvaluator::writeToCopy(FILE* fp, TIndi& indi, std::vector<int> &sol){
	Array.resize(Ncity);
	int curr=0, st=0, count=0, pre=-1, next;
	while( 1 ){
		Array[ count] = curr;
		count++;
		if( count > Ncity ){
			printf( "Invalid\n" );
			return;
		}
		if( indi.fLink[ curr ][ 0 ] == pre ) next = indi.fLink[ curr ][ 1 ];
		else next = indi.fLink[ curr ][ 0 ];

		pre = curr;
		curr = next;
		if( curr == st ) break;
	}

	fprintf( fp, "%d\n\n", indi.fN);
	for( int i = 0; i < indi.fN; ++i )
		fprintf( fp, "%d\n", Array[ i ] );

	sol.assign(Array.begin(), Array.end());
}

void TEvaluator::write2Array(TIndi& indi, int *finalSol){

	int curr=0, st=0, count=0, pre=-1, next;
	while( 1 ){
		finalSol[ count] = curr;
		count++;
		if( count > Ncity ){
			printf( "Invalid\n" );
			return;
		}
		if( indi.fLink[ curr ][ 0 ] == pre ) next = indi.fLink[ curr ][ 1 ];
		else next = indi.fLink[ curr ][ 0 ];

		pre = curr;
		curr = next;
		if( curr == st ) break;
	}
}

bool TEvaluator::checkValid(vector<int>& array, Type_dis value) {
	int *check=new int[Ncity];
	for( int i = 0; i < Ncity; ++i ) check[ i ] = 0;
	for( int i = 0; i < Ncity; ++i ) ++check[ array[ i ]];
	for( int i = 0; i < Ncity; ++i )
		if( check[ i ] != 1 ) return false;
	Type_dis distance = 0;
	for( int i = 0; i < Ncity-1; ++i ){
		distance += fEdgeDis[ array[ i ]][ array[ i+1 ]];
//		std::cout<<i<<" "<<array[i]<<" "<<array[i+1]<<" "<<fEdgeDis[ array[ i ]][ array[ i+1 ]]<<std::endl;
	}

	distance += fEdgeDis[ array[ Ncity-1 ] ][ array[ 0 ]];
//	std::cout<<Ncity-1<<" "<<array[Ncity-1]<<" "<<array[0]<<" "<<fEdgeDis[ array[ Ncity-1 ] ][ array[ 0 ]]<<std::endl;
//	for(int i=0;i<Ncity;++i){
//		for(int j=0;j<Ncity;++j){
//			std::cout<<fEdgeDis[i][j]<<" ";
//		}
//		std::cout<<std::endl;
//	}

	std::cout<<"evaluation value:"<<distance<<std::endl;

	delete [] check;
	if( distance != value ) return false;
	return true;
}

