# meningitis/simulation.py

import numpy as np
import sciris as sc
import starsim as ss
from .people import People
from .model import Meningitis
from .interventions import Vaccine, Treatment
import pandas as pd
import matplotlib.pyplot as plt

def make_sim(seed=1, n_timesteps=50, use_vaccine=False, timestep=10, prob=0.5, imm_boost=2.0):
    age_data = pd.read_csv('data/nigeria_age.csv')
    pars = dict(
        n_agents=2000,
        start=0,
        end=n_timesteps,
        dt=1.0,
        verbose=0,
        rand_seed=seed,
        networks='random',
        diseases=dict(
            type='meningitis',
            waning=0.009,
        )
    )

    people = People(n_agents=2000, age_data=age_data)

    if use_vaccine:
        vaccine = Vaccine(timestep=timestep, prob=prob, imm_boost=imm_boost)
        sim = ss.Sim(pars, people=people, interventions=vaccine)
    else:
        sim = ss.Sim(pars, people=people)

    return sim

def vac_prob(probs=[0.3, 0.5]):
    for prob in probs:
        n_seeds = 20
        n_timesteps = 100
        baseline_results = np.empty((n_seeds, n_timesteps + 1))
        vaccine_results = np.empty((n_seeds, n_timesteps + 1))
        difference_results = np.empty(n_seeds)
        baseline_sims = []
        vaccine_sims = []

        for seed in range(n_seeds):
            baseline_sim = make_sim(seed=seed, n_timesteps=n_timesteps)
            vaccine_sim = make_sim(seed=seed, prob=prob, n_timesteps=n_timesteps, use_vaccine=True)
            baseline_sims.append(baseline_sim)
            vaccine_sims.append(vaccine_sim)
        
        def run_sim(sim):
            sim.run()
            results = sc.objdict()
            results.time = sim.yearvec
            results.n_infected = sim.results.meningitis.n_infected
            return results
        
        baseline_sim_results = sc.parallelize(run_sim, baseline_sims)
        vaccine_sim_results = sc.parallelize(run_sim, vaccine_sims)
        
        for seed in range(n_seeds):
            baseline = baseline_sim_results[seed]
            vaccine = vaccine_sim_results[seed]
            baseline_results[seed, :] = baseline.n_infected
            vaccine_results[seed, :] = vaccine.n_infected
            difference_results[seed] = baseline_results[seed, :].sum() - vaccine_results[seed, :].sum()
        
        lower_bound_baseline = np.quantile(baseline_results, 0.05, axis=0)
        median_baseline = np.quantile(baseline_results, 0.5, axis=0)
        upper_bound_baseline = np.quantile(baseline_results, 0.95, axis=0)
        lower_bound_vaccine = np.quantile(vaccine_results, 0.05, axis=0)
        median_vaccine = np.quantile(vaccine_results, 0.5, axis=0)
        upper_bound_vaccine = np.quantile(vaccine_results, 0.95, axis=0)
        
        time = baseline.time
        lower_bound_diff = np.quantile(difference_results, 0.05)
        upper_bound_diff = np.quantile(difference_results, 0.95)
        median_diff = np.quantile(difference_results, 0.5)
        xx = prob * 100
        title = f'Estimated impact: {median_diff:.0f} (90% CI: {lower_bound_diff:.0f}, {upper_bound_diff:.0f}) infections averted (Prob: {xx}%)'
        
        plt.figure()
        plt.title(title)
        plt.fill_between(time, lower_bound_baseline, upper_bound_baseline, alpha=0.5)
        plt.plot(time, median_baseline, label='Baseline')
        plt.fill_between(time, lower_bound_vaccine, upper_bound_vaccine, alpha=0.5)
        plt.plot(time, median_vaccine, label='With vaccine')
        plt.xlabel('Time')
        plt.ylabel('Number of people infected')
        plt.legend()
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.savefig(f'figs/vaccine_whole_pop{xx}.png')
        plt.show()
