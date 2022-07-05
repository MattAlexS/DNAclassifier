Explanation of Approach

I decided to take the approach of solving this problem with an evolutionary 
algorithm very much like the those described in the Topics in 
Bioinformatics class. In particular the Side Effect Machine with Ring
Optimization seemed very fitting and so I did my best to replicate that.

Since ultimately, the task is still a classification task. I decided to use
the Logistic Regression Classifier in the SciKit Learn python package as
my fitness function. For every Side Effect Machine that I generate, I use
the outputs to train a model using 3/4 of the training data provided, 
reserving 1/4 of the data for Cross Validation (I repeat this 4 times, 
shuffling the sequences each time to eliminate sampling issues). The precentage 
of correct classification in the Cross Validation group accounts for half of
the fitness. Initially, this was my only metric for fitness and was shown to
be ineffective because it left virtually no room for the machines to improve
through evolution, almost every SEM generated with enough states to solve the
problem started with very close to maximum fitness. Most of these machines gave
a solution in which there were 100 sequence from Group 1 and 0 sequences from
Group 2. While this is possibly the truth (and from where I am right now I
have no way of knowing), I found something peculiar.  Every so often, maybe
10% of the time, I would generate a machine that not only had high fitness,
but also grouped the sequences equally (50 of each) into the 4 groups. It
occurred to me that many of the machines may be good for the task, but few
would be exceptional. This hunch that the (100,0,50,50) machines were
slipping through my fitness function was confirmed when I began looking at 
low performing machines. I found that many machines with comparably low
 (.6/1.0) fitness when given the test set also returned (100,0,50,50)
 even though they were performing poorly on the cross validation. In order
to get around this I added another component to the fitness equation, which
assessed the variability of number of sequences in each group. I used this
to punish machines that output highly variable groupings. Unfortunately, this
imposes some unverified outside stipulation onto the machines and none of
the things that lead me to believe this (exceptional machines will be
more rare, some bad machines output highly variable groupings) are conclusive.
It is totally possible that the sequences I have labelled as group 2 are 
actually an outgroup that is closely related to group 1. Without more data to 
train the SEMs on I will not know.

To replicate the ring structure described in Ashlock 2008. I started the initial
random population off in the middle of the ring and limitted how far the
'individuals' could look for a mate as well as how far a child could move
from its parent.  To simplify things I limited the mating to one mating per
generation. I realize that this is not ideal as many of the machines may
be overwritten before they have a chance to mate. To combat this, I added
a tunable parameter of muation to constantly introduce a little randomness
to keep from getting trapped in local maximums. Despite these efforts, this
is still possible.

After simulating the ring space for enough generations. Almost all of the
machines have extremely high fitness. To officially group the sequences, I run
the test data through each machine in the ring, train them on the training set,
and get them to predict the group of the sequences in the test data. I do this
for each machine and they 'vote' for the sequence's group. The highest group is
selected as well as a confidence measure indicating what ratio of the machines
shared this classification 'opinion' for each sequence.

A sample output can be found in output.txt. Since these are random, running
the program again will not generate the same results, but that answer for the
sequence groupings should still bee the same. 
Warning: Sometimes the program gives an error because not all of the spaces
in the ring have a machine in it. If this is the case it is probably best
to run it again with more generations, in order to properly fill the ring so the
machines can transition from exploration to exploitation.

