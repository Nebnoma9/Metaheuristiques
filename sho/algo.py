########################################################################
# Algorithms
########################################################################
import numpy as np
from sklearn.mixture import GaussianMixture

def random(func, init, again):
    """Iterative random search template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    print("best_sol",best_sol)
    i = 0
    while again(i, best_val, best_sol):
        sol = init()
        val = func(sol)
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def greedy(func, init, neighb, again):
    """Iterative randomized greedy heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    print("*********",best_sol)
    i = 1
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol

# TODO add a simulated-annealing template.

def recuit_simule(func, init, neighb, again):
    """Iterative randomized simulated-annealing heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 1
    T = 100
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val or np.random.uniform(0,1) <= np.exp(-(1/T)*(val - best_val)):
            best_val = val
            best_sol = sol
        T = 0.1*T
        i += 1
    return best_val, best_sol

# TODO add a population-based stochastic heuristic template.

def genetic_alg(func, init, mutation, again, size=20):

    list_sol = [init()]*size
    func_vals  = [func(x) for x in list_sol]
    best_sol = list_sol[0]
    best_val = func_vals[0]
    
    i = 1
    while again(i, best_val, best_sol):
        
        parents = select(list_sol,func_vals)
        offsprings = crossover(parents)
        children = mutation(offsprings)
        children_vals = func(children)

        list_sol = replace(list_sol, parents, children, func)
        func_vals = [func(x) for x in list_sol]

        sols_vals = [(list_sol[i], func_vals[i]) for i in range(len(list_sol))]
        sols_vals = sorted(sols_vals, key=lambda item: item[1], reverse=True)
        best_sol = sols_vals[0][0]
        best_val = sols_vals[0][1]

        i+=1
    return best_val, best_sol

def select(sols,sol_vals):
    vals = [(sols[i],sol_vals[i]) for i in range(len(sols))]
    vals = sorted(vals, key=lambda item:item[1], reverse=True)
    selection = vals[0]
    return selection 


def crossover(parents):
    parent = parents[0]
    offsprings = []
    for i in range(len(parent)-1):
        p1 = parent[i]
        p2 = parent[i+1]
        offsprings.append(np.random.uniform(low=p1,
                                            high=p2))
    return offsprings
    
def replace(list_sol, parents, children, func):
    l = [parents[0],children]
    l = [(x,func(x)) for x in l]
    l = sorted(l, key=lambda item: item[1], reverse=True)
    list_sol = [x for x in list_sol if (x!=parents[0]).any()]
    list_sol.append(l[0][0])
    return list_sol


    
"""

def genetic_alg_EDA_bit(func, init, mutation, again, size=1):
    list_sol = [init()]*size
    func_vals  = [func(x) for x in list_sol]
    best_sol = list_sol[0]
    best_val = func_vals[0]
    
    #gmm = GaussianMixture(n_components = 2)
    #gmm.fit(best_sol)
    #print("*************###o",gmm.means_)
    #distribution = np.sum()
    #print('###"',np.sum(list_sol, axis=1))
    print('###')
    i = 1
    while again(i, best_val, best_sol):
        
        parents = select(list_sol,func_vals)
        offsprings = crossover(parents)
        children = mutation(offsprings)
        #print("*************###ertret",np.array(children))
        children_vals = func(children)

        list_sol = replace(list_sol, parents, children,  func)
        func_vals = [func(x) for x in list_sol]

        sols_vals = [(list_sol[i], func_vals[i]) for i in range(len(list_sol))]
        sols_vals = sorted(sols_vals, key=lambda item: item[1], reverse=True)
        best_sol = sols_vals[0][0]
        best_val = sols_vals[0][1]

        i+=1
    return best_val, best_sol


"""
def genetic_EDA_bit(func, init, neighb, again, pop_size=10):
    pop_x = np.array([init() for i in range(pop_size)])
    pop = [func(x) for x in pop_x]

    distribution = np.sum(pop_x+1, axis = 0)/np.sum(pop_x+1)
    best_sol = pop_x[0]
    best_val = pop[0]

    i = 1
    while again(i, best_val, best_sol): #TODO Це хуйня
        samples = []
        for k in range(2*pop_size):
            samples.append(generation_EDA(distribution, 3)) # TODO : nb_sensors

        pop_x = select_EDA(samples, pop_size, func)
        pop = [func(x) for x in pop_x]

        distribution = np.sum(np.array(pop_x)+1, axis = 0)/np.sum(np.array(pop_x)+1)

        sols_vals = [(pop_x[i], pop[i]) for i in range(len(pop))]
        sols_vals = sorted(sols_vals, key=lambda item: item[1], reverse=True)
        best_sol = sols_vals[0][0]
        best_val = sols_vals[0][1]

        i += 1
    return best_val, best_sol


def generation_EDA(distribution, nb_sensors):
    result = np.zeros(distribution.shape)
    k = 0
    while k < nb_sensors:
        rnd = np.random.uniform(0, 1)
        summ_precedent = 0
        summ_curr = 0
        for i in range(distribution.shape[0]):
            for j in range(distribution.shape[1]):
                summ_curr += distribution[i][j]
                if summ_precedent <= rnd <= summ_curr:
                    if result[i][j] == 1:
                        k -= 1
                    result[i][j] = 1
                    k += 1
                summ_precedent += distribution[i][j]
    return result


def select_EDA(samples, pop_size, func):
    pop_x = samples
    pop = [func(x) for x in pop_x]

    vals = [(pop_x[i], pop[i]) for i in range(len(pop))]
    vals = sorted(vals, key=lambda item: item[1], reverse=True)

    new_pop = []
    for l in range(len(vals)):
        new_pop.append(vals[l][0])
    return new_pop[:pop_size]
