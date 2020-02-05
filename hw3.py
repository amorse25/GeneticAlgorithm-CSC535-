# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:51:36 2019

@author: Adam Morse
"""

"""
Dynamic Programming solution to the jump-It problem
The solution finds the cheapest cost to play the game along with the path leading
to the cheapest cost
"""
import random
import numpy as np
global cost, path
sample_set = [1,0]      # sample set to populate population
crossover_rate = 0.75   # crossover rate probability
mutation_rate = 0.002   # mutation rate probability
termination = False


#chromosomes has to be 1 or 0, where 1 = space landed, 0 = skipped


""" Function to create initial population. Uses list structure sample_set
to determine possible values to create population with (1, 0). """
def get_chromosomes(length, board, n_chromes):
    final_chromes = []          # holds the initial population of chromes
    for i in range(n_chromes):
        chromes = []
        while len(chromes) < length:
            sample_size = min(length - len(chromes), len(sample_set))       # randomly generate each chrome
            chromes.extend(random.sample(sample_set, sample_size))          # randomly generate each chrome
        final_chromes.append(chromes) 
    return final_chromes

""" Funtion to fix repeating 0's if found in the initial population. 
Two 0's in a row is not a valid board, as that means two indicies were
skipped. This function replaces the first 0 found for a set of 0's with a 1"""
def fix_repeating_zeros(length, chromes):
    for k in range(len(chromes)):
        chromes[k][0] = 1       # start position at 0 for a hit (1)
        chromes[k][length - 1] = 1 # end at last index
        previous = chromes[k][0]    # previous index
        index = 1
        for num in chromes[k][1:]:
            if num == previous and num == 0:    # if two 0's are found
                chromes[k][index] = 1           # replace one 0 with a 1
            previous = num
            index += 1
    return chromes


""" Function to determine the cost of playing the game based on the random
placement of 1's and 0's for each chromosome. Takes input as the genotype (1's,0's)
and outputs the phenotype (actual values based on the game board) """
def get_chrome_index(index, board):
    board_index = []
    for i in range(len(index)):
        if(index[i] == 1):          # if index in the chrome is a 1 (meaning a hit on the board)
            board_index.append(board[i])    # grab the value (phenotype version)
    return board_index

""" Function to determine the total cost of a chromosome playing the game.
(sum of all values it hit). """
def get_total_cost(chromes):
    total_cost = sum(chromes)       # sum the total values for a total cost
    return total_cost

""" Function to determine the fitness of the chromosomes within the population.
This function improves as the chromosomes total cost decreases. The assumption 
made here is that chromes need to become closer to target. Target in this instance
is the cost of the Dynamic Programing solution playing the game. The closer each 
chromes are to the target cost, means they are improving each generation. When each
chrome has a fitness distance of 0, that means they all have an ideal board solution. """
def get_fitness(parents, board):
    target = jumpIt(board)      # target minimum cost of playing the game (DP solution)
    total_dist = []
    total_sum = 0
    for i in parents:
        dist = (i - target)     # determine the distance each chrome is from the target
        total_dist.append(dist)
    #total_dist.sort()
    for k in total_dist:        # grab the total sum of all chromes (cost - targetcost)
        total_sum += k

    if total_sum == 0:          # when total cost is 0 (meaning all chromes have reached
                                # the target value) termination condition is met. 
        termination = True
        return termination
    
    for j in range(len(total_dist)):
        total_dist[j] = 1 - ((total_dist[j] / total_sum) / 1)   # format values to depict their space for the wheel
        
    fitness_sum = sum(total_dist)
    return total_dist, fitness_sum
    
""" Function to determine what chromes will be selected based on the method
of roulette wheel selection. Each chromes occcupies x-amount of space on the 
wheel. This space is determined by its fitness, meaning the better fitness 
a chrome has, the more space it occupies on the wheel, giving it a better chance
to become selected. This also gives all chromes the chance of being selected, no
matter how poor their fitness score is. """
def get_selection(fitness, n_chromes):     # roulette wheel of selction
    selection = dict()
    possible_selection = dict()
    selected = dict()   # dictionary for the selected chromes (2)
    range_arr = []      # holds the range for each chrome (space)
    prob_arr = []       # holds probability ranges
    length = n_chromes - 1  #length of prob_arr array
    probability = fitness[1]
    prob_end = int(round(fitness[1]))
    rand_num = []
    for k in range(2):      # iterate twice because we need two parents
        rand_num.append(random.uniform(0.0, prob_end))  # generate random number (roullete ball)
   
        for i in range(len(fitness[0])):
            possible_selection[i] = (fitness[0][i])     # determine possible selctions
        
        for key in (possible_selection):
            range_arr.append(probability)               # identify each chromes range (space)
            key_range = probability - possible_selection[key]   # increment each space based on the last iteration
            probability = key_range
            prob_arr.append(probability)    

        for i in range(len(fitness[0])):        
            for key in possible_selection:
                prob_arr[length] = 0.0                 # last chrome should always be > 0 
                selection[i] = (range_arr[i], prob_arr[i])  #selection range for each chrome (space on board)
        for i in range(len(rand_num)):
            for key in selection:
                if rand_num[i] < selection[key][0]:     # make the selection based on random number generated
                    selected[i] = ((key, selection[key]))
    return selected

""" Function to select two parents from the population. The selection
is initially made in get_selection(), this function basically just gives 
a more desired format for later use. """
def selected_parent(chromes, selection):
    parent_select = []
    parents = []
    for i in range(len(selection)):
        parent_select.append(selection[i][0])   # grab the index of the two selected parents from the dictionary
    for index in parent_select:
        parents.append(chromes[index])          # result format back to genotype ([1's, 0's])
    return parents

""" Function to determine if a crossover will take place with both parents.
The possible rate is 0.75 / 1.0, meaning any random number (0.0 - 1.0) selected
that falls <= 0.75 will result in a crossover taking place. The crossover point, 
or index (position on the board) is chosen at random. If no crossover takes place, 
the function just returns a clone of the parents initially passed in. """
def crossover(parents):
    children = []
    offspring = []
    crossover_random = random.random()
    crossover_point = random.randint(1, len(parents[0])-2) # crossover point random number generated.
                    # represents the index at which to split at, and cannot split at the first index, or the last

    if crossover_random <= crossover_rate:      # if crossover should occur (crossover_random <= 0.75)
        for i in range(len(parents)):
            children.append(parents[i][crossover_point:])   # grab the values beyond the crossover point 
            offspring.append(parents[i][:crossover_point])

        for i in range(len(children[0])):
            offspring[0].append(children[1][i])             # swap values 
            offspring[1].append(children[0][i])

        for k in range(len(offspring)):         # if the crossover results in repeating 0's
            previous = offspring[k][0]          
            index = 1
            for num in offspring[k][1:]:
                if num == previous and num == 0:    # replace one 0 with a 1
                    offspring[k][index] = 1
                previous = num
                index += 1
    else:                               # else, clone the parents (no crossover)
        offspring = parents
    return offspring

""" Function to determine if a mutation will take place for the offspring. The mutaation 
rate is 0.002 / 1.0, meaning any random number (0.0 - 1.0) selected that falls <= 0.002
will result in a mutation. """
def mutate(offspring):
    mutate_rand = random.random()       # random number to determine mutation
    if mutate_rand <= mutation_rate:    # if mutate_rand is <= 0.002
        for i in range(len(offspring[0])):
            if offspring[0][i] == 0:    # grab the first 0 in the chrome, and change to a 1
                offspring[0][i] = 1
            if offspring[1][i] == 0:    # Do for both offspring ^
                offspring[1][i] = 1 
    return offspring

""" Function to determine the most fit chromes within the population. The goal is
to determine the two least fit, so they can be removed and replaced with the 
generated offspring. """
def get_most_fit(offspring, final_chromes, board):
    total_cost_chromes = []
    index_to_remove = []
    for i in range(len(final_chromes)):
        final_chromes_cost = get_chrome_index(final_chromes[i], board)  # determine chromes cost of playing
        total_cost_chromes.append(get_total_cost(final_chromes_cost))

    max_cost = max(total_cost_chromes)          # return the max cost out of all chromes (worst player)
    for i in range(len(total_cost_chromes)):
        if total_cost_chromes[i] == max_cost:
            index_to_remove.append(i)           # retrieve the index of the worst player

    if (len(index_to_remove) >= 2):             # if two or more instances of max cost were found 
                                                # (multiple bad players resulted in the same max cost)
        del total_cost_chromes[index_to_remove[0]]  # begin deletion from the population (first chrome)
        del total_cost_chromes[index_to_remove[1]-1] # begin deletion from the population (second chrome)
        del final_chromes[index_to_remove[0]]
        del final_chromes[index_to_remove[1]-1]
        
    if (len(index_to_remove) == 1):             # if only one chrome had the max cost 
        del total_cost_chromes[index_to_remove[0]]  # remove from population and total cost list
        del final_chromes[index_to_remove[0]]
        max_cost = max(total_cost_chromes)          # now that removed, determine new worst cost
        for i in range(len(total_cost_chromes)):
            if total_cost_chromes[i] == max_cost:
                index_to_remove.append(i)           # determine the new index of new worst player
        del total_cost_chromes[index_to_remove[1]]  # remove from the population and total cost list
        del final_chromes[index_to_remove[1]]
        
    for i in range(len(offspring)):
        final_chromes.append(offspring[i])      #append offspring to chromes to create new population
        
    return final_chromes



cost = [] # global table to cache results - cost[i] stores minimum cost of playing the game starting at cell i
path = [] #global table to store path leading to cheapest cost
def jumpIt(board):
    #Bottom up dynamic programming implementation
    #board - list with cost associated with visiting each cell
    #return minimum total cost of playing game starting at cell 0
    
    n = len(board)
    cost[n - 1] = board[n - 1] #cost if starting at last cell
    path[n - 1] = -1 # special marker indicating end of path "destination/last cell reached"
    cost[n - 2] = board[n - 2] + board[n - 1] #cost if starting at cell before last cell
    path[n -2] = n - 1 #from cell before last, move into last cell
    #now fill the rest of the table
    for i in range(n-3, -1, -1):
        #cost[i] = board[i] + min(cost[i+1], cost[i+2])
        if cost[i +  1] < cost[i + 2]: # case it is cheaper to move to adjacent cell
            cost[i] = board[i] +  cost[i + 1]
            path[i] = i + 1 #so from cell i, one moves to adjacent cell
        else: 
            cost[i] = board[i] + cost[i + 2]
            path[i] = i + 2 #so from cell i, one jumps over cell
    return cost[0]

def displayPath(board):
    #Display path leading to cheapest cost - method displays indices of cells visited
    #path - global list where path[i] indicates the cell to move to from cell i
    cell = 0 # start path at cell 0
    print("path showing indices of visited cells:", end = " ")
    print(0, end ="")
    path_contents = "0" # cost of starting/1st cell is 0; used for easier tracing
    while path[cell] != -1: # -1 indicates that destination/last cell has been reached
        print(" ->", path[cell], end = "")
        cell = path[cell]
        path_contents += " -> " + str(board[cell])
    print()
    print("path showing contents of visited cells:", path_contents)

    
def main():
    accuracy =[]
    board_counter = 0
    f = open("input.txt", "r") #input.txt
    global cost, path
    for line in f:
        lyst = line.split() # tokenize input line, it also removes EOL marker
        lyst = list(map(int, lyst))
        cost = [0] * len(lyst) #create the cache table
        path = cost[:] # create a table for path that is identical to path
        min_cost = jumpIt(lyst)
        
        index_path = []         # holds path of traveled index for GA golution
        n_chromes = 3 * len(lyst)       # population size
        generations = 15 * n_chromes# generation size
        board_counter += 1          # determines how many game boards in text file (used for accuracy counting)  
        
        chromosomes = get_chromosomes(len(lyst), lyst, n_chromes)   # generate initial population
        final_chromes = fix_repeating_zeros(len(lyst), chromosomes) # fix repeating 0's in population

        index = 0           # stopping critera for generations
        while index <= generations:     # if index exceeds total possible generation
            #print("GENERATION: ", index)
            total_cost = []
            for i in range(len(final_chromes)):
                board_values = get_chrome_index(final_chromes[i], lyst) # grab board values (cost of playing)
                total_cost.append(get_total_cost(board_values))         # total cost (sum)
            fitness = get_fitness(total_cost, lyst)                     # fitness sum
            if fitness == True:     # Termination criteria if all chromes have reached ideal fitness
                accuracy.append(1)  # hold accuracy score
                break
            else:
                selection = get_selection(fitness, n_chromes)   # select parents
                parents = selected_parent(final_chromes, selection) # parents
                cross_offspring = crossover(parents)        # offspring
                mutation = mutate(cross_offspring)          # mutatations or clones
                most_fit = get_most_fit(mutation, final_chromes, lyst)  # most fit of the population
                final_chromes = most_fit
                index += 1      # increment index
        
        """ DP solution prints """
        print("game board:", lyst)
        print("___________________________")
        print("DP Solution")
        print("minimum cost: ", min_cost)
        displayPath(lyst)
        print("___________________________")
        
        """ GA solution prints """
        print("GA Solution")
        print("Minimum cost(fitness): ", max(total_cost))   # grab max cost (could == min_cost, or the highest (worst performing
                                                             # chrome in the population))
        for i in range(len(final_chromes[0])):
            if final_chromes[0][i] == 1:
                index_path.append(i)
        print("path showing indices of visited cells:", end = " ")
        for i in index_path:
            print(i, "->", end = " ")
        print()
        print("path showing contents of visited cells:", end = " ")
        for i in board_values:
            print(i, "->", end = " ")
        print()
        print("___________end of board: ", board_counter, "___________")
        
    print('GA overall accuracy: ', repr((len(accuracy) * 100.0 ) /float(board_counter))  + '%')
if __name__ == "__main__":
    main()


