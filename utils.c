#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "utils.h"

static uint32_t state = 0xDEADBEEF;

extern dual potential_energy(dual *params);
extern int NUM_PARAMS;

void set_seed(uint32_t s) {
    state = s;
    if (state == 0) state = 0xDEADBEEF;
}

uint32_t xorshift32() {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

double runif(double l_bound, double u_bound) {
    uint32_t val = xorshift32();
    double interval_size = u_bound - l_bound;
    double r = l_bound + interval_size * (double)val / (4294967296.0); 
    return r;
}

double rnorm() {
    static double spare_value;
    static int has_spare = 0;

    if (has_spare) {
        has_spare = 0;
        return spare_value;
    }

    double u1, u2, s;
    do {
        u1 = runif(-1, 1);
        u2 = runif(-1, 1);
        s = (u1 * u1) + (u2 * u2);
    } while ((s == 0) || (s >= 1));

    double multiplier = sqrt((-2 * log(s) / s));
    spare_value = u2 * multiplier;
    has_spare = 1;

    return u1 * multiplier;
}

double rt_df2() {
    double z = rnorm();
    double u = runif(0, 1);
    if (u < 1e-10) u = 1e-10;
    double chi_sq = -2.0 * log(u);
    
    return z / sqrt(chi_sq / 2.0);
}

double qnorm(double p, double mu, double sigma) {
    if (p <= 0 || p >= 1.0) return 0.0;
    double a[] = {
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
    };
    double b[] = {
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01
    };
    double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
    };
    double d[] = {
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00
    };

    double p_low = 0.02425;
    double p_high = 1.0 - p_low;

    double q, r, x;
    if (p < p_low) {
        q = sqrt(-2*log(p));
        x =  (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1));
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        x =  ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
             (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    } else {
        q = sqrt(-2*log(1-p));
        x =  -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1));
    }

    return (mu + sigma * x);  
}

double dpois(int x, double lambda) {
    return (exp(-lambda) * pow(lambda, x) / tgamma(x+1));
}

double log_dpois(double x, double lambda) {
    if (lambda <= 0) return -INFINITY; 
    return (-lambda + x * log(lambda) - lgamma(x+1));
}

double max(double arr[], int n) {
    double biggest = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > biggest) {
            biggest = arr[i];
        }
    }
    return biggest;
}

double sum_arr(double arr[], int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

double log_sum_exp(double a, double b) {
    double max_val = (a > b) ? a : b;
    return max_val + log(1.0 + exp(-fabs(a - b)));
}

double log_sum_exp_arr(double arr[], int n) {
    double biggest = max(arr, n);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += exp(arr[i] - biggest);
    }
    return log(sum) + biggest;
}

int min_index(double arr[], int n) {
    int index = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] < arr[index]) {index = i;}
    }
    return index;
}

double get_norm(double arr[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i] * arr[i];
    }

    return sqrt(sum);
}

double get_norm_centred(double arr[], double coords[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += pow((arr[i] - coords[i]), 2);
    }

    return sqrt(sum);
}

CSVdata load_history(const char *filename) {

    FILE *f = fopen(filename, "r");
    if (!f) { printf("Error opening file\n"); exit(1); }

    // 1. Count lines and columns
    int rows = 0;
    int cols = 0;
    char buffer[16384]; // Big buffer for line reading
    
    // Read header to count columns
    if (fgets(buffer, sizeof(buffer), f)) {
        char *token = strtok(buffer, ",");
        while (token) {
            cols++;
            token = strtok(NULL, ",");
        }
    }

    // Count rows
    while (fgets(buffer, sizeof(buffer), f)) {
        rows++;
    }
    
    // 2. Allocate Memory
    CSVdata hist;
    hist.rows = rows;
    hist.cols = cols;
    hist.data = malloc(rows * sizeof(double*));
    hist.last_col = malloc(rows * sizeof(double));

    // 3. Read Data
    rewind(f);
    fgets(buffer, sizeof(buffer), f); // Skip header again

    int r = 0;
    while (fgets(buffer, sizeof(buffer), f) && r < rows) {
        hist.data[r] = malloc(cols * sizeof(double));
        
        char *token = strtok(buffer, ",");
        int c = 0;
        while (token && c < cols) {
            double val = atof(token);
            hist.data[r][c] = val;
            
            // If this is the last column, save to last_col
            if (c == cols - 1) {
                hist.last_col[r] = val;
            }
            
            token = strtok(NULL, ",");
            c++;
        }
        r++;
    }
    fclose(f);
    printf("Loaded %d samples with %d columns.\n", rows, cols);
    return hist;
}


dual make_dual(double val, double dot) {
    dual result;
    result.val = val;
    result.dot = dot;
    return result;
}

dual make_const(double a) {
    dual result;
    result.val = a;
    result.dot = 0;
    return result;
}

dual add(dual a, dual b) {
    dual result;
    result.val = a.val + b.val;
    result.dot = a.dot + b.dot;
    return result;
}

dual sub(dual a, dual b) {
    dual result;
    result.val = a.val - b.val;
    result.dot = a.dot - b.dot;
    return result;
}

dual mult(dual a, dual b) {
    dual result;
    result.val = a.val * b.val;
    result.dot = b.dot * a.val + a.dot * b.val;
    return result;
}

dual div_ad(dual a, dual b) {
    dual result;
    result.val = a.val / b.val;
    result.dot = (b.val * a.dot - a.val * b.dot) / (b.val * b.val);
    return result;
}

dual sin_ad(dual a) {
    dual result;
    result.val = sin(a.val);
    result.dot = a.dot * cos(a.val);
    return result;
}

dual cos_ad(dual a) {
    dual result;
    result.val = cos(a.val);
    result.dot = a.dot * -sin(a.val);
    return result;
}

dual log_ad(dual a) {
    dual result;
    result.val = log(a.val);
    result.dot = a.dot / a.val;
    return result;
}

dual exp_ad(dual a) {
    double expo = exp(a.val);
    dual result;
    result.val = expo;
    result.dot = a.dot * expo;
    return result;
}

dual pow_ad(dual a, double b) {
    dual result;
    result.val = pow(a.val, b);
    result.dot = b * pow(a.val, b-1) * a.dot;
    return result;
}

dual pow_ad_gen(dual a, dual b) {
    dual result;
    result.val = pow(a.val, b.val);
    double term1 = b.val * pow(a.val, b.val-1) * a.dot;
    double term2 = log(a.val) * pow(a.val, b.val) * b.dot;
    result.dot = term1 + term2;
    return result;
}

dual log_sum_exp_ad(dual a, dual b) {
    dual max_val = (a.val > b.val) ? a : b;
    dual min_val = (a.val <= b.val) ? a : b;
    dual log_inside = add(make_const(1.0), exp_ad(sub(min_val, max_val)));
    return add(max_val, log_ad(log_inside));
}

void print_dual(dual a) {
    printf("Value: %f, derivative: %f\n", a.val, a.dot);
}

double pe_wrapper(double params[]) {
    dual params_ad[NUM_PARAMS];
    for (int i = 0; i < NUM_PARAMS; i++) {
        params_ad[i] = make_const(params[i]);
    }
    dual result = potential_energy(params_ad);
    return result.val;
}

void gradient_U_wrapper(double params[], double grads[]) {
    dual params_ad[NUM_PARAMS];

    for (int i = 0; i < NUM_PARAMS; i++) {
        for (int j = 0; j < NUM_PARAMS; j++) {
            params_ad[j].val = params[j];
            if (i == j) {
                params_ad[j].dot = 1.0;
            } else {
                params_ad[j].dot = 0.0;
            }
        }
        dual result = potential_energy(params_ad);
        grads[i] = result.dot;
    }
}