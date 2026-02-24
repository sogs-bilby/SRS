#ifndef MODEL_H
#define MODEL_H

#include "utils.h"

extern int NUM_PARAMS;

// double log_likelihood(double params[]);
// dual potential_energy(dual params[]);

// void us_to_params(double us[], double *params);

// for nested sampling
double log_likelihood(double params[]);

// for HMC
dual potential_energy(dual params[]);

// NS priors
void us_to_params(double us[], double params[]);

// for HMC
double pe_wrapper(double params[]);
void gradient_U_wrapper(double params[], double grads[]);


#endif