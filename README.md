# DNA Classifier

The purpose of the DNA Classifier is to evolve Side Effect Machines (SEMs) to extact informative features for classifying DNA sequences. 


## Description

This classifier uses SEMs to parse DNA sequences to extract features. 
SEMs are simple finite sate machines that take DNA as input and output 
counts of how many times each state was eneterd. The state information 
is normalized by the length of the DNA sequence, transforming the seuence 
into numerical features. SEMs are evolved in a ring structure to select machines 
that produce useful features. The fitness of an SEM is found by the ability of its
features to train a logistic regression classifier to predict the correct DNA 
class label. 

Since ultimately, the task is still a classification task. I decided to use
the Logistic Regression Classifier in the SciKit Learn python package as
my fitness function. For every Side Effect Machine that I generate, I use
the outputs to train a model using 3/4 of the training data provided, 
reserving 1/4 of the data for Cross Validation (I repeat this 4 times, 
shuffling the sequences each time to eliminate sampling issues). The precentage 
of correct classification in the Cross Validation group gives the fitness. 

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


## Getting Started


### Dependencies

* Python Version 3.0 or higher
    * scikit-learn package

### Installing

Local Installation

1. Git Hub
```git clone https://github.com/MattAlexS/DNAClassifer```

2. Download Zip File
```wget  https://github.com/MattAlexS/DNAClassifer/archive/master.zip```


## Sample Output

A sample output can be found in [output.txt.](/output.txt) 

Since these are random, running
the program again will not generate the same results, but that answer for the
sequence groupings should still be the same. 

## Help

Common issue is that the program gives an error if not all of the spaces
in the ring have a machine in it. If this is the case it is probably best
to run it again with more generations, in order to properly fill the ring so the
machines can transition from exploration to exploitation.


## Authors

Matthew Alexander Stoodley - m.a.stoodley@gmail.com

## Version History

* 0.1
    * Initial Release

## Acknowledgments

My advisor Dr. Daniel Ashlock 
