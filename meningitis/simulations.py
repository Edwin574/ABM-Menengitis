import numpy as np
import sciris as sc

from utils import make_vacc_sim_age,make_age_specific_vacc_sim,make_treatment_sim
### Try different vaccination proportions
def vac_prob(probs = [0.3, 0.5]):
    for prob in probs:
        # Prepare to run multiple times
        n_seeds = 20  # Don't use too many here or your sim will take a very long time!
        n_timesteps = 100  # Again, don't use too many
        baseline_results = np.empty((n_seeds, n_timesteps+1))  # Initialize storage of baseline results
        vaccine_results  = np.empty((n_seeds, n_timesteps+1))  # Initialize storage of vaccine results
        difference_results = np.empty(n_seeds)  # Initialize storage of differences - this will tell us the impact
        baseline_sims = [] # Initialize the list of baseline simulations
        vaccine_sims = [] # Initialize the list of baseline simulations
        # Make the simulations with different seeds
        for seed in range(n_seeds): # Run over 5 different random seeds
            baseline_sim = make_age_specific_vacc_sim(seed=seed, n_timesteps=n_timesteps) # Run the simulation with no vaccine
            vaccine_sim  = make_age_specific_vacc_sim(seed=seed, prob=prob, n_timesteps=n_timesteps, use_vaccine=True) # Run the simulation with the vaccine
            baseline_sims.append(baseline_sim) # Add the baseline sim to the list
            vaccine_sims.append(vaccine_sim) # Add the vaccine sim to the list
        
        def run_sim(sim):
            """ Run the simulation and return the results """
            sim.run()
            results = sc.objdict()
            results.time = sim.yearvec
            results.n_infected = sim.results.meningitis.n_infected
            sim.export_df().to_csv(f"data/meningitis_dynamics_vaccine_{prob}.csv", index=False)
            return results
        
        # Run the simulations in parallel
        baseline_sim_results = sc.parallelize(run_sim, baseline_sims) # Run baseline sims
        vaccine_sim_results  = sc.parallelize(run_sim, vaccine_sims) # Run the vaccine sims
        
        
        # Pull out the results
        for seed in range(n_seeds):
            baseline = baseline_sim_results[seed]
            vaccine = vaccine_sim_results[seed]
            baseline_results[seed, :] = baseline.n_infected # Pull out results from baseline
            vaccine_results[seed, :] = vaccine.n_infected  # Pull out results from vaccine scenarios
            difference_results[seed] = baseline_results[seed, :].sum() - vaccine_results[seed, :].sum()  # Calculate differences
        
        # Get the qunatiles for plotting
        lower_bound_baseline = np.quantile(baseline_results, 0.05, axis=0)
        median_baseline      = np.quantile(baseline_results, 0.5, axis=0)
        upper_bound_baseline = np.quantile(baseline_results, 0.95, axis=0)
        lower_bound_vaccine  = np.quantile(vaccine_results, 0.05, axis=0)
        median_vaccine       = np.quantile(vaccine_results, 0.5, axis=0)
        upper_bound_vaccine  = np.quantile(vaccine_results, 0.95, axis=0)
        
        # Get the time vector for plotting
        time = baseline.time
        
        # Calculate differences
        lower_bound_diff = np.quantile(difference_results, 0.05)
        upper_bound_diff = np.quantile(difference_results, 0.95)
        median_diff = np.quantile(difference_results, 0.5)
        xx = prob*100
        # title = f'Estimated impact: {median_diff:.0f} (90% CI: {lower_bound_diff:.0f}, {upper_bound_diff:.0f}) infections averted (Prob: {xx}%)'      
       
vac_prob()

### Try different vaccination proportions
def vac_prob_age(probs =  [0.5, 1]):
    for prob in probs:
        # Prepare to run multiple times
        n_seeds = 20  # Don't use too many here or your sim will take a very long time!
        n_timesteps = 100  # Again, don't use too many
        age_range = [0.75, 1.5]
        baseline_results = np.empty((n_seeds, n_timesteps+1))  # Initialize storage of baseline results
        vaccine_results  = np.empty((n_seeds, n_timesteps+1))  # Initialize storage of vaccine results
        difference_results = np.empty(n_seeds)  # Initialize storage of differences - this will tell us the impact
        baseline_sims = [] # Initialize the list of baseline simulations
        vaccine_sims = [] # Initialize the list of baseline simulations
        # Make the simulations with different seeds
        for seed in range(n_seeds): # Run over 5 different random seeds
            baseline_sim = make_vacc_sim_age(seed=seed, n_timesteps=n_timesteps) # Run the simulation with no vaccine
            vaccine_sim  = make_vacc_sim_age(seed=seed, age_range=age_range, prob=prob, n_timesteps=n_timesteps, use_vaccine=True) # Run the simulation with the vaccine
            baseline_sims.append(baseline_sim) # Add the baseline sim to the list
            vaccine_sims.append(vaccine_sim) # Add the vaccine sim to the list
        
        def run_sim(sim):
            """ Run the simulation and return the results """
            sim.run()
            results = sc.objdict()
            results.time = sim.yearvec
            results.n_infected = sim.results.meningitis.n_infected
            sim.export_df().to_csv(f"data_output/meningitis_dynamics_vaccine_age_{prob}.csv", index=False)
            return results
        
        # Run the simulations in parallel
        baseline_sim_results = sc.parallelize(run_sim, baseline_sims, serial=False) # Run baseline sims
        vaccine_sim_results  = sc.parallelize(run_sim, vaccine_sims, serial=False) # Run the vaccine sims
        
        
        # Pull out the results
        for seed in range(n_seeds):
            baseline = baseline_sim_results[seed]
            vaccine = vaccine_sim_results[seed]
            baseline_results[seed, :] = baseline.n_infected # Pull out results from baseline
            vaccine_results[seed, :] = vaccine.n_infected  # Pull out results from vaccine scenarios
            difference_results[seed] = baseline_results[seed, :].sum() - vaccine_results[seed, :].sum()  # Calculate differences
        
        # Get the qunatiles for plotting
        lower_bound_baseline = np.quantile(baseline_results, 0.05, axis=0)
        median_baseline      = np.quantile(baseline_results, 0.5, axis=0)
        upper_bound_baseline = np.quantile(baseline_results, 0.95, axis=0)
        lower_bound_vaccine  = np.quantile(vaccine_results, 0.05, axis=0)
        median_vaccine       = np.quantile(vaccine_results, 0.5, axis=0)
        upper_bound_vaccine  = np.quantile(vaccine_results, 0.95, axis=0)
        
        # Get the time vector for plotting
        time = baseline.time
        
        # Calculate differences
        lower_bound_diff = np.quantile(difference_results, 0.05)
        upper_bound_diff = np.quantile(difference_results, 0.95)
        median_diff = np.quantile(difference_results, 0.5)
        xx = prob*100
        # title = f'Estimated impact: {median_diff:.0f} (90% CI: {lower_bound_diff:.0f}, {upper_bound_diff:.0f}) infections averted (Prob: {xx}%)'
vac_prob_age()

def treat_prob(probs = [0.3, 0.5, 0.6, 0.8, 1]):
    for prob in probs:
        # Prepare to run multiple times
        n_seeds = 20  # Don't use too many here or your sim will take a very long time!
        n_timesteps = 100  # Again, don't use too many
        baseline_results = np.empty((n_seeds, n_timesteps+1))  # Initialize storage of baseline results
        treatment_results  = np.empty((n_seeds, n_timesteps+1))  # Initialize storage of treatment results
        difference_results = np.empty(n_seeds)  # Initialize storage of differences - this will tell us the impact
        baseline_sims = [] # Initialize the list of baseline simulations
        treatment_sims = [] # Initialize the list of baseline simulations
        
        # Make the simulations with different seeds
        for seed in range(n_seeds): # Run over 5 different random seeds
            baseline_sim = make_treatment_sim(seed=seed, n_timesteps=n_timesteps) # Run the simulation with no treatment
            treatment_sim  = make_treatment_sim(seed=seed, prob=prob, n_timesteps=n_timesteps, use_treatment=True) # Run the simulation with the treatment
            baseline_sims.append(baseline_sim) # Add the baseline sim to the list
            treatment_sims.append(treatment_sim) # Add the treatment sim to the list
        
        def run_sim(sim):
            """ Run the simulation and return the results """
            sim.run()
            results = sc.objdict()
            results.time = sim.yearvec
            results.n_infected = sim.results.meningitis.n_infected
            sim.export_df().to_csv(f"data/meningitis_dynamics_treatment_{prob}.csv", index=False)
            
            return results
        
        # Run the simulations in parallel
        baseline_sim_results = sc.parallelize(run_sim, baseline_sims) # Run baseline sims
        treatment_sim_results  = sc.parallelize(run_sim, treatment_sims) # Run the treatment sims
        
        
        # Pull out the results
        for seed in range(n_seeds):
            baseline = baseline_sim_results[seed]
            treatment = treatment_sim_results[seed]
            baseline_results[seed, :] = baseline.n_infected # Pull out results from baseline
            treatment_results[seed, :] = treatment.n_infected  # Pull out results from treatment scenarios
            difference_results[seed] = baseline_results[seed, :].sum() - treatment_results[seed, :].sum()  # Calculate differences
        
        # Get the qunatiles for plotting
        lower_bound_baseline = np.quantile(baseline_results, 0.05, axis=0)
        median_baseline      = np.quantile(baseline_results, 0.5, axis=0)
        upper_bound_baseline = np.quantile(baseline_results, 0.95, axis=0)
        lower_bound_treatment  = np.quantile(treatment_results, 0.05, axis=0)
        median_treatment       = np.quantile(treatment_results, 0.5, axis=0)
        upper_bound_treatment  = np.quantile(treatment_results, 0.95, axis=0)
        
        # Get the time vector for plotting
        time = baseline.time
        
        # Calculate differences
        lower_bound_diff = np.quantile(difference_results, 0.05)
        upper_bound_diff = np.quantile(difference_results, 0.95)
        median_diff = np.quantile(difference_results, 0.5)
        # title = f'Estimated impact: {median_diff:.0f} (90% CI: {lower_bound_diff:.0f}, {upper_bound_diff:.0f}) infections averted (Prob: {prob})'
        
        
        # Do the plotting
       
treat_prob()