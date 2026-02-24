#ifndef UTILS_H
#define UTILS_H
#include <stdint.h>

void set_seed(uint32_t s);

uint32_t xorshift32();
double runif(double l_bound, double u_bound);
double rnorm();
double rt_df2();
double qnorm(double p, double mu, double sigma);
double dpois(int x, double lambda);
double log_dpois(double x, double lambda);
double max(double arr[], int n);
double sum_arr(double arr[], int n);
double log_sum_exp(double a, double b);
double log_sum_exp_arr(double arr[], int n);
int min_index(double arr[], int n);
double get_norm(double arr[], int n);
double get_norm_centred(double arr[], double centre[], int n);

typedef struct {
    int rows;
    int cols;
    double **data; // data[row][col]
    double *last_col; // separate shortcut to the last column
} CSVdata;

CSVdata load_history(const char *filename);

typedef struct {
    double val;
    double dot;
} dual;

dual make_dual(double val, double dot);
dual make_const(double a);
dual add(dual a, dual b);
dual sub(dual a, dual b);
dual mult(dual a, dual b);
dual div_ad(dual a, dual b);
dual sin_ad(dual a);
dual cos_ad(dual a);
dual log_ad(dual a);
dual exp_ad(dual a);
dual pow_ad(dual a, double b);
dual pow_ad_gen(dual a, dual b);
dual log_sum_exp_ad(dual a, dual b);
dual log_sum_exp_ad(dual a, dual b);
void print_dual(dual a);

double pe_wrapper(double params[]);
void gradient_U_wrapper(double params[], double grads[]);

#endif