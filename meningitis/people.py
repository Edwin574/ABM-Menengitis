import numpy as np
import pandas as pd
import starsim as ss

class People(ss.People):
    def __init__(self, n_agents=2000, age_data=None, asfr_data=None, deaths_data=None, *args, **kwargs):
        super().__init__(n_agents=n_agents, *args, **kwargs)
        self.age_data = age_data
        self.asfr_data = asfr_data
        self.deaths_data = deaths_data
        self.age = self.initialize_age_distribution(n_agents)
        self.uid = np.arange(n_agents)

    def initialize_age_distribution(self, n_agents):
        if self.age_data is not None:
            age_probs = self.age_data['value'] / self.age_data['value'].sum()
            ages = np.random.choice(self.age_data['age'], size=n_agents, p=age_probs)
        else:
            ages = np.zeros(n_agents)
        return ages
    
    def update_post(self, sim):
        self.age[self.alive] += 1/365
        return self.age

    def apply_births(self, sim):
        if self.asfr_data is None:
            return
        
        current_year = sim.yearvec[sim.ti]
        current_asfr = self.asfr_data[self.asfr_data['Time'] == current_year]
        
        for index, row in current_asfr.iterrows():
            age_grp = row['AgeGrp']
            births = row['Births']
            age_cond = (self.age == age_grp) & self.alive
            
            if births > 0:
                num_births = int(births * age_cond.sum())
                new_ages = np.zeros(num_births)
                self.add_agents(num_births, new_ages)
                print(f"Added {num_births} new agents")

    def apply_deaths(self, sim):
        if self.deaths_data is None:
            return
        
        current_year = sim.yearvec[sim.ti]
        current_deaths = self.deaths_data[self.deaths_data['Time'] == current_year]
        
        for index, row in current_deaths.iterrows():
            age_start = row['AgeGrpStart']
            mortality_rate = row['mx']
            age_cond = (self.age >= age_start) & self.alive
            
            if mortality_rate > 0:
                num_deaths = int(mortality_rate * age_cond.sum())
                death_indices = np.random.choice(np.where(age_cond)[0], num_deaths, replace=False)
                self.alive[death_indices] = False
                print(f"Applied {num_deaths} deaths")

    def add_agents(self, num, ages):
        self.n_agents += num
        self.age = np.concatenate((self.age, ages))
        self.alive = np.concatenate((self.alive, np.ones(num, dtype=bool)))
        new_uids = np.arange(self.uid.max() + 1, self.uid.max() + 1 + num)
        self.uid = np.concatenate((self.uid, new_uids))
        print(f"Updated uid array to {self.uid.shape[0]} agents")
