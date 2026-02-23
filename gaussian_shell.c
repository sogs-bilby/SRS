#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "model.h"
#include "utils.h"

int NUM_PARAMS = 2;

// ===================
// FOR NS
// ===================

double log_likelihood(double params[]) {
    // gaussian shell

    double coords1[NUM_PARAMS];
    double coords2[] = {-4.0, 0.0};

    for (int i=0; i < NUM_PARAMS; i++) {
        coords1[0] = 4.0;
        coords1[1] = 0.0;
    }

    double width = 0.1;
    double radius = 1.0;
    double denominator = 2 * (width * width);

    double numerator1 = pow(get_norm_centred(params, coords1, NUM_PARAMS) - radius, 2);
    double frac1 = numerator1 / denominator;

    double numerator2 = pow(get_norm_centred(params, coords2, NUM_PARAMS) - radius, 2);
    double frac2 = numerator2 / denominator;

    // return -frac1;
    
    return log_sum_exp(-frac1, -frac2); 
}

void us_to_params(double us[], double *params) {
    for (int i = 0; i < NUM_PARAMS; i++) {
        params[i] = -6.0 + 12.0 * us[i];
        // params[i] = 0.5 + 1.0 * us[i];
        // params[i] = -1.0 + 5.0 * us[i];
    }
}




// ===================
// FOR HMC
// ===================

// double get_norm(double arr[], int n) {
//     double sum = 0;
//     for (int i = 0; i < n; i++) {
//         sum += arr[i] * arr[i];
//     }

//     return sqrt(sum);
// }

// dual get_norm_ad(dual params[], int n)

dual potential_energy(dual params[]) {
    // gaussian shell

    double radius = 1.0;
    double width = 0.1;
    

    dual x = params[0];
    dual y = params[1];
    dual x_pos = pow_ad(sub(x, make_const(4.0)), 2);
    dual x_neg = pow_ad(add(x, make_const(4.0)), 2);

    dual denominator = mult(make_const(2.0), pow_ad(make_const(width), 2.0));

    dual numerator1 = pow_ad(
                            sub(
                                pow_ad(
                                    add(x_pos, 
                                        mult(
                                            y, y)
                                        ), 0.5
                                    ), make_const(radius)
                            ), 2.0
                        );

    dual numerator2 = pow_ad(
                            sub(
                                pow_ad(
                                    add(x_neg, 
                                        mult(
                                            y, y)
                                        ), 0.5
                                    ), make_const(radius)
                            ), 2.0
                        );

    dual frac1 = mult(make_const(-1.0), div_ad(numerator1, denominator));
    dual frac2 = mult(make_const(-1.0), div_ad(numerator2, denominator));

    

    return mult(make_const(-1.0), log_sum_exp_ad(frac1, frac2));
}