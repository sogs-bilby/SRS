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
    // unwrapped multi D rosenbrock

    double sum = 0;

    for (int i = 0; i < NUM_PARAMS-1; i++) {
        double term1 = 100 * pow((params[i+1] - pow(params[i], 2)), 2);
        double term2 = pow(1-params[i], 2);
        sum += term1 + term2;
    }
    
    return -sum; 
}

void us_to_params(double us[], double *params) {
    for (int i = 0; i < NUM_PARAMS; i++) {
        // params[i] = 0.55 + 0.1 * us[i];
        // params[i] = 0.5 + 1.0 * us[i];
        params[i] = -6.0 + 12.0 * us[i];
        // params[i] = 0.5;
    }
}




// ===================
// FOR HMC
// ===================

dual potential_energy(dual params[]) {
    // unchained rosenbrock
    dual sum = make_const(0.0);

    for (int i = 0; i < NUM_PARAMS-1; i++) {
        dual term1 = sub(params[i+1], mult(params[i], params[i]));
        dual term2 = pow_ad(sub(make_const(1.0), params[i]), 2.0);
        dual term3 = mult(make_const(100.0), mult(term1, term1));
        sum = add(sum, add(term2, term3));
    }
    
    return sum;
}