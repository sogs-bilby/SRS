#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "utils.h"
#include "model.h"


#define PRIOR_MIN -6.0
#define PRIOR_MAX 6.0
#define PROPOSAL_SIGMA 0.5

double log_prior(double *params) {
    for (int i = 0; i < NUM_PARAMS; i++) {
        if (params[i] < PRIOR_MIN || params[i] > PRIOR_MAX) {
            return -INFINITY; 
        }
    }
    return 0.0;
}

double log_posterior(double *params) {
    double lp = log_prior(params);
    if (lp == -INFINITY) return -INFINITY;
    
    double ll = log_likelihood(params);
    return lp + ll;
}


void generate_proposal(double *dest, double *src) {
    memcpy(dest, src, NUM_PARAMS * sizeof(double));
    for (int i = 0; i < NUM_PARAMS; i++) {
        dest[i] += rnorm() * PROPOSAL_SIGMA;
    }
}

int main(int argc, char* argv[]) {
    uint32_t seed = (uint32_t)(time(NULL) ^ (uintptr_t)malloc(1));
    set_seed(seed);

    int n_iter = 100000;
    int thin = 10;
    
    if (argc >= 2) n_iter = atoi(argv[1]);
    if (argc >= 3) thin = atoi(argv[2]);

    printf("Running Metropolis for %d iterations (Thinning: %d)...\n", n_iter, thin);
    printf("Dimensions: %d | Step Size: %.3f\n", NUM_PARAMS, PROPOSAL_SIGMA);

    double *current_params = malloc(NUM_PARAMS * sizeof(double));
    double *proposal_params = malloc(NUM_PARAMS * sizeof(double));

    for (int i = 0; i < NUM_PARAMS; i++) {
        current_params[i] = runif(PRIOR_MIN, PRIOR_MAX);
    }

    double current_logp = log_posterior(current_params);
    
    FILE *fptr = fopen("met_output.csv", "w");
    if (!fptr) { perror("Error opening file"); return 1; }
    
    for (int i = 1; i < NUM_PARAMS+1; i++) {
        fprintf(fptr, "V%d,", i);
    }
    fprintf(fptr, "log_prob\n"); 

    int accepted = 0;
    
    for (int iteration = 1; iteration <= n_iter; iteration++) {
        
        generate_proposal(proposal_params, current_params);
        double proposal_logp = log_posterior(proposal_params);

        double log_alpha = proposal_logp - current_logp;
        if (isnan(log_alpha)) log_alpha = -INFINITY;

        if (log(runif(0, 1)) < log_alpha) {
            memcpy(current_params, proposal_params, NUM_PARAMS * sizeof(double));
            current_logp = proposal_logp;
            accepted++;
        }

        if (iteration % thin == 0) {
            for (int i = 0; i < NUM_PARAMS; i++) {
                fprintf(fptr, "%f,", current_params[i]);
            }
            fprintf(fptr, "%f\n", current_logp);
        }

        if (iteration % (n_iter / 10) == 0) {
            double rate = (double)accepted / iteration;
            printf("Iter: %d | Acc Rate: %.3f | LogP: %.2f\n", iteration, rate, current_logp);
        }
    }

    fclose(fptr);
    free(current_params);
    free(proposal_params);
    printf("Done. Saved to met_output.csv\n");

    return 0;
}