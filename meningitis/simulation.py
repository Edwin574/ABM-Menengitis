
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
    asfr_data = pd.read_csv('data/nigeria_asfr.csv')
    deaths_data = pd.read_csv('data/nigeria_deaths.csv')
    pars = dict(
        n_agents=2000,
        start=0,
        end=n_timesteps,
        dt=1.0,
        verbose=0,
        rand_seed=seed,
        networks='random'
    )

    people = People(n_agents=2000, age_data=age_data, asfr_data=asfr_data, deaths_data=deaths_data)
    meningitis = Meningitis()

    if use_vaccine:
        vaccine = Vaccine(timestep=timestep, prob=prob, imm_boost=imm_boost)
        sim = ss.Sim(pars, people=people, diseases=meningitis, interventions=vaccine)
    else:
        sim = ss.Sim(pars, people=people, diseases=meningitis)

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
        
        # plt.figure()
        # plt.title(title)
        # plt.fill_between(time, lower_bound_baseline, upper_bound_baseline, alpha=0.5)
        # plt.plot(time, median_baseline, label='Baseline')
        # plt.fill_between(time, lower_bound_vaccine, upper_bound_vaccine, alpha=0.5)
        # plt.plot(time, median_vaccine, label='With vaccine')
        # plt.xlabel('Time')
        # plt.ylabel('Number of people infected')
        # plt.legend()
        # plt.ylim(bottom=0)
        # plt.xlim(left=0)
        # plt.savefig(f'figs/vaccine_whole_pop{xx}.png')
        # plt.show()

### Additional Simulations with Age-Specific Data

def vac_prob_age(probs=[0.5, 1], age_range=[0.75, 1.5]):
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
        
        # plt.figure()
        # plt.title(title)
        # plt.fill_between(time, lower_bound_baseline, upper_bound_baseline, alpha=0.5)
        # plt.plot(time, median_baseline, label='Baseline')
        # plt.fill_between(time, lower_bound_vaccine, upper_bound_vaccine, alpha=0.5)
        # plt.plot(time, median_vaccine, label='With vaccine')
        # plt.xlabel('Time')
        # plt.ylabel('Number of people infected')
        # plt.legend()
        # plt.ylim(bottom=0)
        # plt.xlim(left=0)
        # plt.savefig(f'figs/vaccine_age{xx}.png')
        # plt.show()

def treat_prob(probs=[0.3, 0.5, 0.6, 0.8, 1]):
    for prob in probs:
        n_seeds = 20
        n_timesteps = 100
        baseline_results = np.empty((n_seeds, n_timesteps + 1))
        treatment_results = np.empty((n_seeds, n_timesteps + 1))
        difference_results = np.empty(n_seeds)
        baseline_sims = []
        treatment_sims = []

        for seed in range(n_seeds):
            baseline_sim = make_sim(seed=seed, n_timesteps=n_timesteps)
            treatment_sim = make_sim(seed=seed, prob=prob, n_timesteps=n_timesteps, use_treatment=True)
            baseline_sims.append(baseline_sim)
            treatment_sims.append(treatment_sim)
        
        def run_sim(sim):
            sim.run()
            results = sc.objdict()
            results.time = sim.yearvec
            results.n_infected = sim.results.meningitis.n_infected
            return results
        
        baseline_sim_results = sc.parallelize(run_sim, baseline_sims)
        treatment_sim_results = sc.parallelize(run_sim, treatment_sims)
        
        for seed in range(n_seeds):
            baseline = baseline_sim_results[seed]
            treatment = treatment_sim_results[seed]
            baseline_results[seed, :] = baseline.n_infected
            treatment_results[seed, :] = treatment.n_infected
            difference_results[seed] = baseline_results[seed, :].sum() - treatment_results[seed, :].sum()
        
        lower_bound_baseline = np.quantile(baseline_results, 0.05, axis=0)
        median_baseline = np.quantile(baseline_results, 0.5, axis=0)
        upper_bound_baseline = np.quantile(baseline_results, 0.95, axis=0)
        lower_bound_treatment = np.quantile(treatment_results, 0.05, axis=0)
        median_treatment = np.quantile(treatment_results, 0.5, axis=0)
        upper_bound_treatment = np.quantile(treatment_results, 0.95, axis=0)
        
        time = baseline.time
        lower_bound_diff = np.quantile(difference_results, 0.05)
        upper_bound_diff = np.quantile(difference_results, 0.95)
        median_diff = np.quantile(difference_results, 0.5)
        title = f'Estimated impact: {median_diff:.0f} (90% CI: {lower_bound_diff:.0f}, {upper_bound_diff:.0f}) infections averted (Prob: {prob})'
        
        # plt.figure()
        # plt.title(title)
        # plt.fill_between(time, lower_bound_baseline, upper_bound_baseline, alpha=0.5)
        # plt.plot(time, median_baseline, label='Baseline')
        # plt.fill_between(time, lower_bound_treatment, upper_bound_treatment, alpha=0.5)
        # plt.plot(time, median_treatment, label='With treatment')
        # plt.xlabel('Time')
        # plt.ylabel('Number of people infected')
        # plt.legend()
        # plt.ylim(bottom=0)
        # plt.xlim(left=0)
        # plt.show()

def plot_more(sim, var, add=False, nrow=2, ncol=2, figsize=(8, 8)):
    if (ncol * nrow) > len(var):
        ncol = 1
        nrow = len(var)
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        x = sim.tivec
        if add:
            y = sim.results.meningitis[var[i]]
        else:
            y = sim.results[var[i]]
        ax.plot(x, y)
        ax.set_title(var[i])
    plt.tight_layout()
    return fig
