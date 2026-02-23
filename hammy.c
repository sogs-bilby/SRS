#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "model.h"
#include "utils.h"

int n_iter = 10000;
int thin = 1;

// ==========
// ACTUAL HMC
// ==========

double kinetic_energy(double momentum[]) {
    double k = 0.0;
    for (int i = 0; i < NUM_PARAMS; i++) {
        k += 0.5 * momentum[i] * momentum[i];
    }
    return k;
}

void leapfrog(double *q, double *p, double epsilon, int L, int n_params, double *grad_buffer) {
    gradient_U_wrapper(q, grad_buffer);

    for (int i = 0; i < n_params; i++) {
        p[i] = p[i] - (epsilon / 2.0) * grad_buffer[i];
    }

    for (int l = 0; l < L; l++) {
        for (int i = 0; i < n_params; i++) {
            q[i] = q[i] + epsilon * p[i];
        }

        if (l < L - 1) {
            gradient_U_wrapper(q, grad_buffer);
            for (int i = 0; i < n_params; i++) {
                p[i] = p[i] - epsilon * grad_buffer[i];
            }
        }
    }
    gradient_U_wrapper(q, grad_buffer);
    for (int i = 0; i < n_params; i++) {
        p[i] = p[i] - (epsilon / 2.0) * grad_buffer[i];
    }
}

int main(int argc, char* argv[]) {
    uint32_t seed = (uint32_t)(time(NULL) ^ (uintptr_t)malloc(1));
    set_seed(seed);

    char parameter_names[NUM_PARAMS][20];
    for (int i = 0; i < NUM_PARAMS; i++) {
        sprintf(parameter_names[i], "V%d", i+1);
    }

    double epsilon = 0.002; 
    int L = 50;
    char *filename = "hmc_output.csv";

    if (argc >= 2) epsilon = atof(argv[1]);
    if (argc >= 3) L = atoi(argv[2]);
    if (argc >= 4) filename = argv[3];

    double *current_q = malloc(NUM_PARAMS * sizeof(double));
    double *current_p = malloc(NUM_PARAMS * sizeof(double));
    double *prop_q = malloc(NUM_PARAMS * sizeof(double));
    double *prop_p = malloc(NUM_PARAMS * sizeof(double));
    double *grad_buffer = malloc(NUM_PARAMS * sizeof(double));


    for (int i = 0; i < NUM_PARAMS; i++) {
        current_q[i] = runif(-6, 6);
        printf("%f\n", current_q[i]);
    }

    FILE *fptr = fopen(filename, "w");
    for (int i = 0; i < NUM_PARAMS; i++) {
        fprintf(fptr, "%s,", parameter_names[i]);
    }
    fprintf(fptr, "potential_energy\n");

    int accepted = 0;
    printf("Starting HMC with eps=%f, L=%d\n", epsilon, L);

    for (int iter = 1; iter <= n_iter; iter++) {
        for (int i = 0; i < NUM_PARAMS; i++){
            current_p[i] = rnorm();
        }

        memcpy(prop_q, current_q, NUM_PARAMS * sizeof(double));
        memcpy(prop_p, current_p, NUM_PARAMS * sizeof(double));

        double current_U = pe_wrapper(current_q);
        double current_K = kinetic_energy(current_p);
        double current_H = current_U + current_K;

        leapfrog(prop_q, prop_p, epsilon, L, NUM_PARAMS, grad_buffer);

        double prop_U = pe_wrapper(prop_q);
        double prop_K = kinetic_energy(prop_p);
        double prop_H = prop_U + prop_K;

        double H_diff = prop_H - current_H;
        
        if (log(runif(0,1)) < -H_diff) {
            memcpy(current_q, prop_q, NUM_PARAMS * sizeof(double));
            accepted++;
        }

        if (iter % thin == 0) {
            for (int i = 0; i < NUM_PARAMS; i++) {
                fprintf(fptr, "%f,", current_q[i]);
            }
            fprintf(fptr, "%f\n", current_U);
            fflush(fptr);
            printf("Iteration %d. Acceptance rate = %f\n", iter, (double)accepted/iter);
        }
    }
    printf("Done. Output saved to %s.\n", filename);

    fclose(fptr);
    free(current_q); free(current_p); free(prop_q); free(prop_p); free(grad_buffer);
    return 0;
}