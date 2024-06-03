from model import Meningitis
import starsim as ss
from people import People
from interventins.vaccination.vaccine import VaccineAge,Vaccine
import pandas as pd

def init_sim_data():
    """This function recieves parameters, performs initial simulation and returns the simulation data in json format"""
    meningitis=Meningitis()
    start=int(input("Enter start year: "))
    end=int(input("Enter end year: "))
    init_population=int(input("Enter initial population: "))
    
    pars = {
    'networks': {'type': 'random'},
    'start': start,
    'end': end,
    'dt': 1,
    'verbose': 0,
    'n_agents': init_population
}
    sim = ss.Sim(pars, diseases=meningitis)
    sim.run()
    initial_data = sim.export_df().to_json()
    return initial_data
    
# data=init_sim_data() 
# print(data) 
# 
def make_age_specific_vacc_sim(seed=1, n_timesteps=50, use_vaccine=True, timestep=10, prob=0.5, imm_boost=2.0):
    """ Make the simulat\ion, but do not run it yet """
    age_data=pd.read_csv("data_input/nigeria_age.csv")
    pars = dict(
        n_agents = 2000,
        start = 0,
        end = n_timesteps,
        dt = 1.0,
        verbose = 0,
        rand_seed = seed,
        networks = 'random',
        diseases = dict(
            type = 'meningitis',
            waning = 0.009,
        )
    )

    people = People(n_agents=2000, age_data=age_data)

    # Define "baseline" and "intervention" sims without and with the vaccine
    if use_vaccine:
        vaccine = Vaccine(timestep=timestep, prob=prob, imm_boost=imm_boost)
        sim = ss.Sim(pars, people=people, interventions=vaccine)
        sim.run()
        # sim.export_df().to_csv("data/meningitis_vaccine_age_data.csv", index=False)
    else:
        sim = ss.Sim(pars, people=people)
        sim.plot()
        

    return sim
def make_vacc_sim_age(seed=1, age_range=None, n_timesteps=100, use_vaccine=False, timestep=10, prob=0.5, imm_boost=2.0):
    """ Make the simulation, but do not run it yet """

    pars = dict(
        start = 0,
        end = n_timesteps,
        dt = 1.0,
        verbose = 0,
        rand_seed = seed,
        networks = 'random',
        diseases = dict(
            type = 'meningitis',
            waning = 0.009,
        )
    )

    people = People(n_agents=2000, age_data=None)

    # Define "baseline" and "intervention" sims without and with the vaccine
    if use_vaccine:
        vaccine = VaccineAge(timestep=timestep, age_range=age_range, prob=prob, imm_boost=imm_boost)
        sim = ss.Sim(pars, people=people, interventions=vaccine)
    else:
        sim = ss.Sim(pars, people=people)

    return sim

def make_treatment_sim(seed=1, n_timesteps=100, use_treatment=False, timestep=20, prob=0.5, mean_dur_infection=5):
    """ Make the simulation, but do not run it yet """

    pars = dict(
        n_agents = 2000,
        start = 0,
        end = n_timesteps,
        dt = 1.0,
        verbose = 0,
        rand_seed = seed,
        networks = 'random',
        diseases = dict(
            type = 'meningitis',
        )
    )

    # Define "baseline" and "intervention" sims without and with the treatment
    if use_treatment:
        treatment = Treatment(timestep=timestep, prob=prob, mean_dur_infection=mean_dur_infection)
        sim = ss.Sim(pars, interventions=treatment)
    else:
        sim = ss.Sim(pars)

    return sim