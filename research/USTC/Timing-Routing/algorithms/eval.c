#include "eval.h"

double eval(double* points, int* res, int degree) {
    double length = 0;
    double xs[degree], ys[degree];
    double xl[degree], xr[degree];
    double yl[degree], yh[degree];
    int i, j;
    for (i = 0, j = 0; i < degree;) {
        xs[i] = points[j++];
        ys[i] = points[j++];
        i++;
    }
    size_t cpy_len = sizeof(double) * degree;
    memcpy(xl, xs, cpy_len);
    memcpy(xr, xs, cpy_len);
    memcpy(yl, ys, cpy_len);
    memcpy(yh, ys, cpy_len);
    int x_idx, y_idx;
    double e_x, e_y, val;
    int limit = degree + degree - 2;
    printf("limit:%d\n", limit);
    for (i = 0; i < limit; ) {
        printf("executing the res %d and %d\n", i, i+1);
        x_idx = res[i++];
        y_idx = res[i++];
        e_x = xs[x_idx];
        e_y = ys[y_idx];
        val = yl[x_idx]; yl[x_idx] = (val < e_y ? val : e_y);
        val = xl[y_idx]; xl[y_idx] = (val < e_x ? val : e_x);
        val = yh[x_idx]; yh[x_idx] = (val > e_y ? val : e_y);
        val = xr[y_idx]; xr[y_idx] = (val > e_x ? val : e_x);
    }
    for (i = 0; i < degree; i++) {
        length += yh[i] - yl[i] + xr[i] - xl[i];
    }
    printf("length:%lf\n", length);
    return length;
}

int call_gst_rsmt(int nterms, double* terms, double* length, int* nsps, double* sps, int* nedges, int* edges) {
    return gst_rsmt(nterms, terms, length, nsps, sps, nedges, edges, NULL, NULL);
}

//int main(){
//    //double a[20] = {0.361856, 0.222882, 0.533107, 0.463135, 0.81636, 0.491875, 0.407179, 0.861488, 0.1027, 0.868141, 0.586552, 0.820974, 0.402049, 0.10883, 0.248102, 0.771204, 0.18469, 0.0337566, 0.365449, 0.529808};
//    double a[20] = {0.529711,0.139518,0.358654,0.102809,0.122585,0.25597,0.487359,0.73978,0.0112047,0.0287552,0.361856, 0.222882, 0.533107, 0.463135, 0.81636, 0.491875, 0.407179, 0.861488, 0.1027, 0.868141};
//    int res[18] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6};
//    printf("eval:%f\n", eval(a, res, 10));
//    return 0;
//}