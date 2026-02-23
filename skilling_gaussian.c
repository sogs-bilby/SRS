#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "model.h"
#include "utils.h"

int NUM_PARAMS = 20;

// ===================
// FOR NS
// ===================

double log_likelihood(double params[]) {
    double u = 0.01;
    double v = 0.1;
    double const_pi_term = -10 * log(2 * M_PI);

    double term1 = const_pi_term - 20 * log(u) + log(100);
    double term2 = const_pi_term - 20 * log(v);

    double sum1 = 0;
    for (int i = 0; i < NUM_PARAMS; i++) {
        sum1 += pow(params[i], 2);
    }

    term1 -= sum1 / (2.0 * u * u);
    term2 -= sum1 / (2.0 * v * v);
    return log_sum_exp(term1, term2);
}

void us_to_params(double us[], double *params) {
    for (int i = 0; i < NUM_PARAMS; i++) {
        // params[i] = -0.1 + 0.2 * us[i];
        params[i] = -0.5 + 1.0 * us[i];
        // params[i] = -1.0 + 5.0 * us[i];
    }
}




// ===================
// FOR HMC
// ===================

dual potential_energy(dual params[]) {
    double u = 0.01;
    double v = 0.1;
    double const_pi_term = -10 * log(2.0 * M_PI);
    
    double const_term1 = const_pi_term - 20.0 * log(u) + log(100.0);
    double const_term2 = const_pi_term - 20.0 * log(v);

    dual term1 = make_const(const_term1);
    dual term2 = make_const(const_term2);

    dual sum1 = make_const(0.0);
    for (int i = 0; i < NUM_PARAMS; i++) {
        sum1 = add(sum1, mult(params[i], params[i]));
    }

    double const_denom1 = 2.0 * u * u;
    double const_denom2 = 2.0 * v * v;

    dual quad1 = div_ad(sum1, make_const(const_denom1));
    dual quad2 = div_ad(sum1, make_const(const_denom2));
    term1 = sub(term1, quad1);
    term2 = sub(term2, quad2);

    return mult(log_sum_exp_ad(term1, term2), make_const(-1.0));
}