#MATTHEW STOODLEY (0988764)
#BINF 6210 - Bioinformatic Software Tools
#Final Assignment - DNA Classification

#Side Effect Machine Evolutionary Algorithm in A Ring Shaped Space
#Given: 4 distinct groups of sequences in synd1-4.fasta
#Goal: To correctly classify the sequences in TestData.fasta into 1 of the 4 groups

#importing necessary libraries
#for generating random numbers

import random

#for machine learning algorithm

from sklearn.linear_model import LogisticRegression

#for counting 'votes'

from collections import Counter



####THE SIDE EFFECT MACHINE CLASS######################################
#this class requires the NUMBER OF STATES to be set
#and a GRAPH describing the relationship of those states.
#A graph takes the form of a dictionary:
#Keys are the states.
#Values are lists of length 4, each element corresponding to
#the next state if that input appears.
# 
#The RUN method of the SEM class takes a sequence as input and outputs
#an n-dimensional vector as output, where n is the number of states.
######################################################################

class SEM:
    
    def __init__(self,statenum, graph):   #initializing the object with input parameters
        self.statenum = statenum
        self.graph = graph
        
    def run(self, sequence):            #starts in states zero and navigates its graph according to
        self.state = 0                  #the input sequence
        self.output = [0]*self.statenum
        for base in sequence:       #iterating through the sequence
            self.output[self.state] += 1  #incrementing the count for each state, starts at state zero
            if base == "A":
                self.state = self.graph[self.state][0]  #chaging state based on the input
            elif base == "C":
                self.state = self.graph[self.state][1]
            elif base == "G":
                self.state = self.graph[self.state][2]
            elif base == "T":
                self.state = self.graph[self.state][3]
        self.output[self.state] += 1    #incrementing the count for the last state
        return self.output              #returns a vector with the counts for each time a state was entered


####THE FITNESS FUNCTION###################################################
#Takes an SEM, training data, and testingdata as inputs
#Outputs a fitness value evaluating the effectiveness of that state machine
#Half of the fitness is the performance of a logistic regression model
#when trained and cross validated on known data.
#The other half is the variability of results when the model is applied
#to the test data (I know this sounds honky, and to some degree it is,
#I elborate more on this in the readme.txt)
###########################################################################
    
def fitness(sem, traindata, testdata):
    assessments = []
    for i in range(4):          #the model is trained and cross validated 4 times
        random.shuffle(traindata)   #each time the order of the sequences is shuffled to change what is trained and validated
        x = []                      
        y = []
        for entry in traindata:     #splitting the training data set into Xs and Ys
            x.append(sem.run(entry[0]))     #Xs are the output of the given SEM for each sequence
            y.append(entry[1])              #Ys are the known sequence groupings (1-4)
        key = y[-150:]           #Reserving the last 50 Ys to cross validate
        #training a logistic regression model with one-vs-rest setting for multiple classifications (1-4)
        #and an inverse regularization coefficient of 1e3
        #first 150 sequences for training, last 50 for cross validation
        #performance variable containes the prediction for the last 50 sequences
        performance = LogisticRegression(C=1e3, multi_class='ovr').fit(x[:-150],y[:-150]).predict(x[-150:])
        count = 0
        for i in range(len(performance)):       #comparing the prediction to the key
            if performance[i] == key[i]:        
                count += 1
        assessments.append(count/len(performance))  #calculating ratio of correct guesses
    averagetraining = (sum(assessments)/4)  #averaging the 4 outcomes
    test_feat = []
    for seq in testdata:
        test_feat.append(sem.run(seq))      #running the test sequences through the SEM
    guess = LogisticRegression(C=1e3, multi_class='ovr').fit(x,y).predict(test_feat) #training a model and predicting the testdata classification
    counts =[0,0,0,0]       #Counting the number of sequences predicted to be in each group 
    for i in guess:
        if i == 1:
            counts[0] += 1
        elif i == 2:
            counts[1] += 1
        elif i == 3:
            counts[2] += 1
        elif i == 4:
            counts[3] += 1
    absolute=[]
    for i in counts:
        absolute.append(abs(50-i))      #calculating the variability of the counts
    testingrecognition = sum(absolute)/300
    return averagetraining - testingrecognition
#fitness is higher when cross validation performs well
#and lower when variation in the sequence counts is high



####FINAL FUNCTION##########################################################
#sorry for the corny name
#Takes a graph for an SEM, the training data, and the unknown data as inputs
#Returns an array where each element is the prediction for that sequence
############################################################################
def momentoftruth(graph, traindata, testdata):
    x = []
    y = []
    test = SEM(len(graph),graph)            #very similar to fitness, no variability
    for entry in traindata:                 #no cross validation, at this point it is 
        x.append(test.run(entry[0]))        #assumed the SEMs are 'good'
        y.append(entry[1])
    test_feat = []
    for seq in testdata:
        test_feat.append(test.run(seq))
    guess = LogisticRegression(C=1e3,multi_class='ovr').fit(x,y).predict(test_feat)
    """
    print(guess)
    one = 0
    two = 0
    three = 0
    four = 0
    for i in guess:
        if i == 1:
            one += 1
        elif i == 2:
            two += 1
        elif i == 3:
            three += 1    #Modified to show more detail for debugging and
        elif i == 4:      #understanding the results
            four += 1
    print("1 ", one)
    print("2 ", two)
    print("3 ", three)
    print("4 ", four)
    """
    return guess #outputs an guesses for each sequence in an array



####THE MATCHMAKER FUNCTION##########################################
#Searches the space where machines can play and selects two mates
#Takes a ring (datingpool) and a distance any given machine
#can travel in the ring as inputs
#returns the graphs of two potential mates as well as the index
#of the first to establish where the offspring can be placed
#####################################################################

def matchmaker(datingpool, materange):
    done = False
    while done == False:
        first = random.randint(0,len(datingpool)-1)     #randomly select a spot in the ring
        if datingpool[first][1] != 0:                   #check if there is a machine there
            done = True
    done = False
    while done == False:                                #looking for the partner
        if first == 0:                                  #making sure not to get stuck looking off the ends of the ring
            second = first + random.randint(1,materange)
        elif first == len(datingpool)-1:
            second = first - random.randint(1,materange)
        else:
            direction = random.randint(0,1)             #coin flip for direction, add random magnitude
            magnitude = random.randint(1,materange)     #within materange
            if direction == 0:
                second = first + magnitude
            else:
                second = first - magnitude
        if second in range(len(datingpool)) and datingpool[second][1] != 0: #still within the ring
            done = True
    match = [datingpool[first][0], datingpool[second][0], first]
    return match


####MATING FUNCTION#################################################
#Produces an offspring from a match, blends two graphs together also
#control the ability to mutate if desired
#Takes a mating pair + the position of the first parent and
#a percent chance of mutation as inputs
#Returns a new graph that is a product of the two mates and the
#position of the first parent
####################################################################

def mate(mates, mutationpercent):
    offspring = {}
    for i in range(len(mates[0])):      #recreates the directions at each state by
        edge = []                       #randomly assigning one of the parents directions
        for j in range(4):
            edge.append(mates[random.randint(0,1)][i][j])
        offspring[i] = edge
    mutate = random.randint(0,99)       #controls chance of mutating
    if mutate < round(mutationpercent): #randomly changes one spot in the graph
        offspring[random.randint(0,len(mates[0])-1)][random.randint(0,3)] = random.randint(0,len(mates[0])-1)
    return [offspring, mates[2]]            


####DELIVERING THE CHILD############################################
#Acknoledging corny name
#Takes an offspring, position of its primary parent, max distance it
#can placed from the parent, the space (datingpool), the training
#data and the test data as inputs
#Selects a viable space to place the child in the ring and replaces
#the current occupant if the child's fitness is higher
####################################################################

def stork(offspring, dispersionrange, datingpool, traindata, testdata):
    score = fitness(SEM(len(offspring[0]),offspring[0]), traindata, testdata) #calculates fitness of offspring
    done = False
    while done == False:        #picking a viable position
        flip = random.randint(0,1)
        if offspring[1] == len(datingpool)-1:
            address = offspring[1] - random.randint(1, dispersionrange)
        elif offspring[1] == 0:
            address = offspring[1] + random.randint(1, dispersionrange)       
        elif flip == 0:
            address = offspring[1] + random.randint(1, dispersionrange)              
        elif flip == 1:
            address = offspring[1] - random.randint(1, dispersionrange)
        if address in range(len(datingpool)):
            done = True
    if datingpool[address][1] < score:         #adds offspring at position only if it has higher fitness
        datingpool[address] = [offspring[0],score]

        
####RANDOM RING INITIATION FUNCTION##################################
#Creates the ring space and generates random SEMs to start the evolution
#Takes parameters of the ring as inputs
#maxpop = ring capacity
#seedsize = number of starting members
#statecount = number of states in a SEM
#training and testing datasets
#Returns a ring with randomly generated individuals starting in the middle
#ready to spread around the sides
#####################################################################

def randominit(maxpop, seedsize, statecount, traindata,testdata):
    seed = []
    community = []
    for i in range(seedsize):       #generating the random machines
        random_machine = {}
        for i in range(statecount):
            edges = []
            edges.append(random.randint(0,statecount-1))
            edges.append(random.randint(0,statecount-1))
            edges.append(random.randint(0,statecount-1))
            edges.append(random.randint(0,statecount-1))
            random_machine[i] = edges
        seed.append(random_machine)
    for i in seed:      #adding the seeds to the community (ring)
        community.append([i, fitness(SEM(statecount, i), traindata,testdata)]) #the community is a list made up lists containing an SEMs graph and its fitness 
    startempty = maxpop - seedsize      
    if startempty % 2 == 0:     #extending the ring on both sides
        for i in range(startempty//2):
            community.append([0,0])
        community.reverse()
        for i in range(startempty//2):
            community.append([0,0])
    else:
        for i in range(startempty//2):
            community.append([0,0])
        community.reverse()
        for i in range(startempty//2 + 1):
            community.append([0,0])  #filling the rest of the ring with empty spaces
    return community


####SIMULATE FUNCTION##################################################
#Combines almost all of the previous function to build a ring structure
#fill it full of random machines and simulate evolution occurring for
#a given number of generations
#Takes many inputs
#Returns the ring structure after the specified number of generations
#In this setup, a generation is just a single mating
#Every time the population size either grows or stays the same
#######################################################################

def simulate(statecount, popsize, seedsize, materange, generations, mutrate, traindata, testdata):
    community = randominit(popsize, seedsize, statecount, traindata, testdata)
    for i in range(generations):
        stork(mate(matchmaker(community, materange), mutrate), materange, community, traindata, testdata)
    return community



#Reading in sequence data
#Generating the datasets in a specific format

file1 = open("synd1.fasta", "r")
file2 = open("synd2.fasta", "r")
file3 = open("synd3.fasta", "r")
file4 = open("synd4.fasta", "r")
file5 = open("TestData.fasta", "r")

traindata = []
testdata = []
testseqnames = []

for line in file1:
    if line[0] != ">":  #throwing away sequence names
        traindata.append([line.strip(), 1])  #appending a list containing the sequence and its group to the list 'data'

        
for line in file2:
    if line[0] != ">":
        traindata.append([line.strip(), 2])


for line in file3:
    if line[0] != ">":
        traindata.append([line.strip(), 3])


for line in file4:
    if line[0] != ">":
        traindata.append([line.strip(), 4])


with open("TestData.fasta","r") as file5:
    for line in file5:
        if line[0] != ">":
            testdata.append(line.strip())       #grabbing the sequences in order
        else:
            testseqnames.append(line.strip()) #grabbing the sequence IDs in order

file1.close()
file2.close()
file3.close()
file4.close()
file5.close()

"""

samplegraph = {
    0: [0, 6, 2, 5],
    1: [1, 3, 4, 1],   #Suprisingly good 7 state SEM used for testing
    2: [4, 0, 0, 3],
    3: [4, 1, 2, 1],
    4: [1, 6, 4, 4],
    5: [2, 6, 2, 3],
    6: [0, 6, 3, 1]
}

"""

#Providing the required output
#Initializing a smallish ring with 20 starting 6 state side effect machines
#Running it for a short 1000 generations, because python
#Using training data and testdata above

statenum = 6
popsize = 80
startsize = 20
matedist = 4
generations = 1300
mutationrate = 100

output = simulate(statenum, popsize, startsize, matedist, generations, mutationrate, traindata, testdata)
runningtotal = 0
count = 0
for i in output:                    #summing all the fitness values to calculate an average fitness
    if i[1] != 0:
        runningtotal += i[1]
        count +=1
print("Ring Size: %d" % popsize)            #printing aspects of the machine
print("Starting Seed size: %d" % startsize)
print("Maximum travel distance: %d" % matedist)
print("Generations: %d" % generations)
print("Mutation rate: %d" % mutationrate)
print("Number of States in a SEM: %d" % statenum)
print("Average SEM Fitness:")
print(runningtotal/count)
print("Sequence Name\tGroup\tConfidence")
vote = []                           #at this point all of the SEMs are very good at the task
for i in range(200):                #setting up a voting system, all the machines will vote for which group
    vote.append([])                 #they thing the sequence belongs too
for i in output:
    tally = momentoftruth(i[0], traindata, testdata)
    for guess in range(200):
        vote[guess].append(tally[guess])
for i in range(len(vote)):
    c = Counter(vote[i])                  #counting up the vote
    print(testseqnames[i], "\t", c.most_common(1)[0][0], "\t", round(c.most_common(1)[0][1]/popsize,2))





