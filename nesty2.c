#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <string.h>
#include <stdbool.h>
#include "utils.h"
#include "model.h"


// #define NUM_PARAMS 3
#define NUM_PARTICLES 1000
#define MCMC_STEPS 1000
#define BUFFER_SIZE 20


// ======================
// ACTUAL NESTED SAMPLING
// ======================
double depth = INFINITY;


typedef struct {
    double log_likelihood;
    int accepted;
} MCMCResult;

// updates particle array in place
MCMCResult do_mcmc(double particle[], double current_logL, double threshold) {
    int accepted = 0;
    double new_particle[NUM_PARAMS];
    double temp_real_params[NUM_PARAMS];
    
    for(int j = 0; j < MCMC_STEPS; j++) {
        // copy current to new
        for(int p=0; p < NUM_PARAMS; p++) new_particle[p] = particle[p];
        
        // how many parameters to perturb
        int num = 1;
        if(runif(0, 1) < 0.5) {
            num = floor(pow(NUM_PARAMS, runif(0, 1)));
        }
        if(num < 1) num = 1;
        
        for(int k=0; k < num; k++) {
            int which = (int)floor(runif(0, NUM_PARAMS));
            if(which == NUM_PARAMS) which = NUM_PARAMS - 1; // boundary check

            double scale = pow(10.0, 1.0 - 3.0 * fabs(rt_df2()));
            
            new_particle[which] += rnorm() * scale;
            
            // force between 0,1
            new_particle[which] -= floor(new_particle[which]);
            if(new_particle[which] < 0) new_particle[which] += 1.0;
        }
        
        // new lik
        us_to_params(new_particle, temp_real_params);
        double logL_new = log_likelihood(temp_real_params);
        
        if(logL_new >= threshold) {
            // accept then update
            for(int p=0; p < NUM_PARAMS; p++) particle[p] = new_particle[p];
            current_logL = logL_new;
            accepted++;
        }
    }
    
    MCMCResult res;
    res.log_likelihood = current_logL;
    res.accepted = accepted;
    return res;
}

int main(int argc, char* argv[]) {
    uint32_t seed = (uint32_t)(time(NULL) ^ (uintptr_t)malloc(1));
    set_seed(seed);

    char parameter_names[NUM_PARAMS][20];
    for (int i = 0; i < NUM_PARAMS; i++) {
        sprintf(parameter_names[i], "V%d", i+1);
    }

    double steps_max = floor(NUM_PARTICLES * depth);
    double log_likelihoods[NUM_PARTICLES];
    double particles[NUM_PARTICLES][NUM_PARAMS];
    double temp_params[NUM_PARAMS];
    char *filename = "ns_output.csv";
    if (argc >= 2) filename = argv[1];

    for (int i = 0; i < NUM_PARTICLES; i++) {
        for (int j = 0; j < NUM_PARAMS; j++) {
            particles[i][j] = runif(0, 1);
        }
        us_to_params(particles[i], temp_params);
        log_likelihoods[i] = log_likelihood(temp_params);
    }

    double best = -INFINITY;
    
    FILE *fptr = fopen(filename, "w");
    for (int i = 0; i < NUM_PARAMS; i++) {
        fprintf(fptr, "%s,", parameter_names[i]);
    }
    fprintf(fptr, "log_lik\n");
    fclose(fptr);

    int iteration = 1;
    int worst_idx;

    MCMCResult res;
    res.accepted = 0;
    
    printf("Starting Nested Sampling...\n");
    while (true) {
        // find worst particle
        worst_idx = min_index(log_likelihoods, NUM_PARTICLES);
        double threshold = log_likelihoods[worst_idx];

        // save worst particle details to file
        fptr = fopen(filename, "a");
        us_to_params(particles[worst_idx], temp_params);
        for (int i = 0; i < NUM_PARAMS; i++) {
            fprintf(fptr, "%f,", temp_params[i]);
        }
        fprintf(fptr, "%f\n", threshold);
        fclose(fptr);

        // update evidence
        double ln_p = threshold - (double)iteration / (double)NUM_PARTICLES;
        if (ln_p > best) best = ln_p;

        if(iteration % NUM_PARTICLES == 0) {
            printf("Iteration %d. LogL: %.4f. Max Weight (approx): %.4f\n", 
                    iteration, threshold, best);
            printf("Acceptance rate: %f\n", (double)res.accepted / (double)MCMC_STEPS);
        }

        // check termination
        bool done = false;
        if (!isinf(steps_max) && iteration >= steps_max) {
            done = true;
        } else if (isinf(depth)) {
            // stop if the remaining volume contributes negligibly to evidence
            if (ln_p < best - log(1e10))
                done = true;
        }

        if (done) {
            printf("Termination condition reached at iteration %d.\n", iteration);
            break;
        }

        // copy random survivor
        int survivor;
        if (NUM_PARTICLES > 1) {
            do {
                survivor = (int)floor(runif(0, NUM_PARTICLES));
                if(survivor == NUM_PARTICLES) survivor = NUM_PARTICLES - 1;
            } while (survivor == worst_idx);
            
            for(int j=0; j<NUM_PARAMS; j++) {
                particles[worst_idx][j] = particles[survivor][j];
            }
            log_likelihoods[worst_idx] = log_likelihoods[survivor];
        }

        // evolve new point
        res = do_mcmc(particles[worst_idx], log_likelihoods[worst_idx], threshold);
        // printf("Acceptance rate: %f\n", (double)res.accepted / (double)MCMC_STEPS);
        log_likelihoods[worst_idx] = res.log_likelihood;

        iteration++;
    }

    printf("Done. Output saved to ns_output.csv\n");

    CSVdata past_data = load_history(filename);
    int N_SAMPLES = past_data.rows;

    double *logws = malloc(N_SAMPLES * sizeof(double));
    double *logws_plus_liks = malloc(N_SAMPLES * sizeof(double));
    double *post_weights = malloc(N_SAMPLES * sizeof(double));
    double H = 0.0;
    double ess_sum = 0.0;

    for (int i = 0; i < N_SAMPLES; i++) {
        logws[i] = -(double)(i+1) / (double)NUM_PARTICLES;
    }
    double logZ_prior = log_sum_exp_arr(logws, N_SAMPLES);
    for (int i = 0; i < N_SAMPLES; i++) {
        logws[i] -= logZ_prior;
        logws_plus_liks[i] = logws[i] + past_data.last_col[i];
    }

    double logZ = log_sum_exp_arr(logws_plus_liks, N_SAMPLES);
    for (int i = 0; i < N_SAMPLES; i++) {
        post_weights[i] = exp(logws_plus_liks[i] - logZ);
        H += post_weights[i] * (past_data.last_col[i] - logZ);
        ess_sum += post_weights[i] * log(post_weights[i] + 1e-300);
    }

    int ess = (int)floor(exp(-ess_sum));
    printf("%f\n", ess_sum);
    printf("%d\n", ess);
    double err = sqrt(H / NUM_PARTICLES);
    double top = max(post_weights, N_SAMPLES);

    printf("DEBUG: LogZ: %f\n", logZ);
    printf("DEBUG: Max Weight found: %f\n", top);
    printf("DEBUG: Sample size (N_SAMPLES): %d\n", N_SAMPLES);

    printf("\nMarginal likelihood: ln(Z) = %f +/- %f\n", logZ, err);
    printf("Information: H = %f nats\n", H);
    printf("Effective posterior sample size = %d\n", ess);


    FILE *f_post = fopen("ns_posterior_samples.csv", "w");
    for (int i = 0; i < NUM_PARAMS; i++) {
        fprintf(f_post, "%s,", parameter_names[i]);
    }
    fprintf(f_post, "log_lik\n");
    printf("Starting resampling...\n");
    
    int k = 0;
    while (k < ess) {
        int which = (int)floor(runif(0, N_SAMPLES));
        double prob = post_weights[which] / top;
        if (runif(0, 1) <= prob) {
            for (int i = 0; i < past_data.cols; i++) {
                fprintf(f_post, "%f", past_data.data[which][i]);
                if (i < (past_data.cols-1)) fprintf(f_post, ",");
            }
            fprintf(f_post, "\n");
            k++;

            if (k % 100 == 0) printf("Resampled %d/%d...\r", k, ess);
        }
    }

    fclose(f_post);
    printf("\nPosterior samples saved to ns_posterior_samples.csv\n");

    free(logws); free(logws_plus_liks); free(post_weights);
    for(int i=0; i<past_data.rows; i++) free(past_data.data[i]);
    free(past_data.data); free(past_data.last_col);

    return 0;
}