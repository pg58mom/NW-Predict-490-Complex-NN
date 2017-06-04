# -*- coding: utf-8 -*-
# We will randomly define initial values for connection weights, and also randomly select
#   which training data that we will use for a given run.
import random
from random import randint

# We want to use the exp function (e to the x); it's part of our transfer function definition
from math import exp

# Biting the bullet and starting to use NumPy for arrays
import numpy as np

# So we can make a separate list from an initial one
import copy

# So we can read in the Grey Box 1 datafiles, which are stored in CSV (comma separated value) format
import csv

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True) 

####################################################################################################
####################################################################################################
#
# This is a tutorial program, designed for those who are learning Python, and specifically using 
#   Python for neural networks applications
#
# It is a multi-component neural network, comprising: 
#  - A "Grey Box," which is subnet with pre-determined weights (read in from data files), that 
#    determines the "big shape class" into which a given input pattern belongs, and
#  - A Convolution Neural Network (CNN) component, currently with a SINGLE (vertical) masking field
#    applied to the original input data (MF1), and retaining the original size of the input data set. 
#    (This is done via expanding the original input data with a one-pixel-width border prior to
#    applying the masking field.)
# The input to the NN is an 81-unit ("pixel") list which represents a 9x9 grid layout for an
#    alphabetic character. 
# This input feeds into both GB1 and the CNN MF1. 
# The network then functions normally with a single hidden layer, where the inputs to the hidden layer 
#   come from Grey Box 1 (GB1) and the result of Masking Field 1 (MF1). 
# The outputs are the set of all 26 (capital) alphabet letters. 
# Ideally, the neural network is trained on both variants and noisy data. This version only provides 
#   a limited (16-character) subset of the inputs, and no variants or noise. 
#
####################################################################################################
####################################################################################################
#
# Code Map: List of Procedures / Functions
# - welcome
#
# == set of basic functions ==
# - computeTransferFnctn
# - computeTransferFnctnDeriv
# - matrixDotProduct
#
# == identify crucial parameters (these can be changed by the user) ==
# - obtainNeuralNetworkSizeSpecs
#    -- initializeWeight
# - initializeWeightArray
# - initializeBiasWeightArray
#
# == obtain data from external data files for GB1
# - readGB1wWeightFile
# - readGB1vWeightFile
# - reconstructGB1wWeightArray
# - reconstructGB1vWeightArray
# - readGB1wBiasWeightFile
# - readGB1vBiasWeightFile
# - reconstructGB1wBiasWeightArray
# - reconstructGB1vBiasWeightArray
#
# == obtain the training data (two possible routes; user selection & random) ==
# - obtainSelectedAlphabetTrainingValues
# - obtainRandomAlphabetTrainingValues
#
# == the feedforward modules ==
#   -- ComputeSingleFeedforwardPassFirstStep
#   -- ComputeSingleFeedforwardPassSecondStep
# - ComputeOutputsAcrossAllTrainingData
#
# == the backpropagation training modules ==
# - backpropagateOutputToHidden
# - backpropagateBiasOutputWeights
# - backpropagateHiddenToInput
# - backpropagateBiasHiddenWeights
#
#
# - main




####################################################################################################
####################################################################################################
#
# Procedure to welcome the user and identify the code
#
####################################################################################################
####################################################################################################


def welcome ():


    print
    print '******************************************************************************'
    print
    print 'Welcome to the Multilayer Perceptron Neural Network'
    print '  trained using the backpropagation method.'
    print 'Version 0.4, 03/05/2017, A.J. Maren'
    print 'For comments, questions, or bug-fixes, contact: alianna.maren@northwestern.edu'
    print ' ' 
    print 'This program learns to distinguish between broad classes of capital letters'
    print 'It allows users to examine the hidden weights to identify learned features'
    print
    print '******************************************************************************'
    print
    return()

        

####################################################################################################
####################################################################################################
#
# A collection of worker-functions, designed to do specific small tasks
#
####################################################################################################
####################################################################################################

   
#------------------------------------------------------#    

# Compute neuron activation using sigmoid transfer function
def computeTransferFnctn(summedNeuronInput, alpha):
    activation = 1.0 / (1.0 + exp(-alpha*summedNeuronInput)) 
    return activation
  

#------------------------------------------------------# 
    
# Compute derivative of transfer function
def computeTransferFnctnDeriv(NeuronOutput, alpha):
    return alpha*NeuronOutput*(1.0 -NeuronOutput)     


#------------------------------------------------------# 
def matrixDotProduct (matrx1,matrx2):
    dotProduct = np.dot(matrx1,matrx2)
    
    return(dotProduct)    


####################################################################################################
####################################################################################################
#
# Function to obtain the neural network size specifications
#
####################################################################################################
####################################################################################################

def obtainNeuralNetworkSizeSpecs ():

# This procedure operates as a function, as it returns a single value (which really is a list of 
#    three values). It is called directly from 'main.'
#        
# This procedure allows the user to specify the size of the input (I), hidden (H), 
#    and output (O) layers.  
# These values will be stored in a list, the arraySizeList. 
# This list will be used to specify the sizes of two different weight arrays:
#   - wWeights; the Input-to-Hidden array, and
#   - vWeights; the Hidden-to-Output array. 
# However, even though we're calling this procedure, we will still hard-code the array sizes for now.   

# Define parameters for the Grey Box 1 (GB1) subnet
    GB1numInputNodes = 121
    GB1numHiddenNodes = 6
    GB1numOutputNodes = 5  

# Define parameters for the full network 
#   This network works with FOUR masking filters, each providing 81 elements from a masking field. 
#     This is a total of 4 x 81 or 324 inputs from masking fields. 
#     There are also 9 inputs (with this current version) from the GB1 outputs. 
#   Thus, there are a total of 324 + 9 = 333 inputs. 
#            
    numInputNodes = 368
    numHiddenNodes = 40
    numOutputNodes = 26 
          
    print ' '
    print '  For the Grey Box 1 subnet, the number of nodes at each level are:'
    print '    Input: 11x11 (square array) = ', GB1numInputNodes
    print '    Hidden: ', GB1numHiddenNodes
    print '    Output: ', GB1numOutputNodes

    print '  For the full multi-component network, the number of nodes at each level are:'
    print '    Input: 11x11 (square array) plus the GB 1 outputs =', numInputNodes
    print '    Hidden: ', numHiddenNodes
    print '    Output: ', numOutputNodes            
                                    
# We create a list containing the crucial SIZES for the connection weight arrays                
    arraySizeList = (GB1numInputNodes, GB1numHiddenNodes, GB1numOutputNodes, numInputNodes, numHiddenNodes, numOutputNodes)
    
# We return this list to the calling procedure, 'main'.       
    return (arraySizeList)  


####################################################################################################
#
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
#
####################################################################################################

def InitializeWeight ():

    randomNum = random.random()
    weight=1-2*randomNum
#    print weight
           
    return (weight)  



####################################################################################################
####################################################################################################
#
# Function to initialize the node-to-node connection weight arrays
#
####################################################################################################
####################################################################################################

def initializeWeightArray (weightArraySizeList):

# This procedure is also called directly from 'main.'
#        
# This procedure takes in the two parameters, the number of nodes on the bottom (of any two layers), 
#   and the number of nodes in the layer just above it. 
#   It will use these two sizes to create a weight array.
# The weights will initially be assigned random values here, and 
#   this array is passed back to the 'main' procedure. 

    
    numLowerNodes = weightArraySizeList[0] 
    numUpperNodes = weightArraySizeList[1] 
      

# Initialize the weight variables with random weights    
    weightArray = np.zeros((numUpperNodes,numLowerNodes))    # iniitalize the weight matrix with 0's
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(numLowerNodes):  # number of columns in matrix 2
            weightArray[row,col] = InitializeWeight ()
                                
# We return the array to the calling procedure, 'main'.       
    return (weightArray)  


####################################################################################################
####################################################################################################
#
# Function to initialize the bias weight arrays
#
####################################################################################################
####################################################################################################

def initializeBiasWeightArray (numBiasNodes):

# This procedure is also called directly from 'main.'

# Initialize the bias weight variables with random weights    
    biasWeightArray = np.zeros(numBiasNodes)    # iniitalize the weight matrix with 0's
    for node in range(numBiasNodes):  #  Number of nodes in bias weight set
        biasWeightArray[node] = InitializeWeight ()
                     
# We return the array to the calling procedure, 'main'.       
    return (biasWeightArray)  




####################################################################################################
####################################################################################################
#
# Function to return a trainingDataList
#
####################################################################################################
####################################################################################################

def obtainSelectedAlphabetTrainingValues (dataSet):
    
########
#  A   #
########

    trainingDataListA0 =  (1,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],0,'A',2,'F') # training data list bottom left 'A' variant 1
    
    trainingDataListA1 =  (27,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,1,0,0,0,0,0, 
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],0,'A',2,'F') # training data list bottom center 'A' variant 1

    trainingDataListA2 =  (28,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,1,0,0,0,0, 
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],0,'A',2,'F') # training data list bottom right 'A' variant 1  
    

    trainingDataListA3 =  (29,[
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    ],0,'A',2,'F') # training data list top left 'A' variant 1
    
    trainingDataListA4 =  (30,[
    0,0,0,0,0,1,0,0,0,0,0, 
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],0,'A',2,'F') # training data list top center 'A' variant 1

    trainingDataListA5 =  (31,[
    0,0,0,0,0,0,1,0,0,0,0, 
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],0,'A',2,'F') # training data list top right 'A' variant 1    
    
           
    trainingDataListA6 =  (32,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],0,'A',2,'F') # training data list bottom left 'A' variant 2
    
    trainingDataListA7 =  (33,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,1,0,0,0,0,0, 
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,    
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],0,'A',2,'F') # training data list bottom center 'A' variant 2

    trainingDataListA8 =  (34,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,1,0,0,0,0, 
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,    
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],0,'A',2,'F') # training data list bottom right 'A' variant 2 
    

    trainingDataListA9 =  (35,[
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,   
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    ],0,'A',2,'F') # training data list top left 'A' variant 2
    
    trainingDataListA10 =  (36,[
    0,0,0,0,0,1,0,0,0,0,0, 
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,    
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],0,'A',2,'F') # training data list top center 'A' variant 2

    trainingDataListA11 =  (37,[
    0,0,0,0,0,0,1,0,0,0,0, 
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,    
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],0,'A',2,'F') # training data list top right 'A' variant 2          
              

########
#  B   #
########                        

    trainingDataListB0 =  (2,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    ],1,'B',0,'L') # training data list bottom left 'B' variant 1
    
    trainingDataListB1 =  (38,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],1,'B',0,'L') # training data list bottom center 'B' variant 1

    trainingDataListB2 =  (39,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],1,'B',0,'L') # training data list bottom right 'B' variant 1  
            
    trainingDataListB3 =  (40,[
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],1,'B',0,'L') # training data list top left 'B' variant 1
    
    trainingDataListB4 =  (41,[
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],1,'B',0,'L') # training data list top center 'B' variant 1

    trainingDataListB5 =  (42,[
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],1,'B',0,'L') # training data list top right 'B' variant 1  
    
    trainingDataListB6 =  (43,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0, 
    ],1,'B',0,'L') # training data list bottom left 'B' variant 2
    
    trainingDataListB7 =  (44,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0, 
    ],1,'B',0,'L') # training data list bottom center 'B' variant 2

    trainingDataListB8 =  (45,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0, 
    ],1,'B',0,'L') # training data list bottom right 'B' variant 2    
    
    trainingDataListB9 =  (46,[
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],1,'B',0,'L') # training data list top left 'B' variant 2
    
    trainingDataListB10 =  (47,[
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],1,'B',0,'L') # training data list top center 'B' variant 2

    trainingDataListB11 =  (48,[
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],1,'B',0,'L') # training data list top right 'B' variant 2   
    
    

########
#  C   #
########                        

    trainingDataListC0 =  (3,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,0,1,1,1,1,1,1,0,0,0, 
    ],2,'C',1,'O')  # training data list bottom left 'C' variant 1
    
    trainingDataListC1 =  (49,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,0,1,1,1,1,1,1,0,0,  
    ],2,'C',1,'O')  # training data list bottom center 'C' variant 1

    trainingDataListC2 =  (50,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,1,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,0,1,1,1,1,1,1,0,
    ],2,'C',1,'O')  # training data list bottom right 'C' variant 1  
            
    trainingDataListC3 =  (51,[
    0,0,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,0,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top left 'C' variant 1
    
    trainingDataListC4 =  (52,[
    0,0,0,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,0,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],2,'C',1,'O')  # training data list top center 'C' variant 1

    trainingDataListC5 =  (53,[
    0,0,0,0,1,1,1,1,1,1,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,0,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top right 'C' variant 1  
    
    trainingDataListC6 =  (54,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],2,'C',1,'O')  # training data list bottom left 'C' variant 2
    
    trainingDataListC7 =  (55,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],2,'C',1,'O')  # training data list bottom center 'C' variant 2

    trainingDataListC8 =  (56,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],2,'C',1,'O')  # training data list bottom right 'C' variant 2  
            
    trainingDataListC9 =  (57,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top left 'C' variant 2
    
    trainingDataListC10 =  (58,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],2,'C',1,'O')  # training data list top center 'C' variant 2

    trainingDataListC11 =  (59,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top right 'C' variant 2 
    
    
########
#  D   #
########                        

    trainingDataListD0 =  (4,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    ],3,'D',1,'O')  # training data list bottom left 'D' variant 1
    
    trainingDataListD1 =  (60,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],3,'D',1,'O')  # training data list bottom center 'D' variant 1

    trainingDataListD2 =  (61,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],3,'D',1,'O')  # training data list bottom right 'D' variant 1      
    
    trainingDataListD3 =  (62,[
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],3,'D',1,'O')  # training data list top left 'D' variant 1
    
    trainingDataListD4 =  (63,[ 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],3,'D',1,'O')  # training data list top center 'D' variant 1

    trainingDataListD5 =  (64,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],3,'D',1,'O')  # training data list top right 'D' variant 1                     
    
    trainingDataListD6 =  (65,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,1,1,0,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    ],3,'D',1,'O')  # training data list bottom left 'D' variant 2
    
    trainingDataListD7 =  (66,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],3,'D',1,'O')  # training data list bottom center 'D' variant 2

    trainingDataListD8 =  (67,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],3,'D',1,'O')  # training data list bottom right 'D' variant 2     
    
    trainingDataListD9 =  (68,[
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,1,1,0,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],3,'D',1,'O')  # training data list top left 'D' variant 2
    
    trainingDataListD10 =  (69,[ 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],3,'D',1,'O')  # training data list top center 'D' variant 2

    trainingDataListD11 =  (70,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],3,'D',1,'O')  # training data list top right 'D' variant 2                     


########
#  E   #
########                        

    trainingDataListE0 =  (5,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0, 
    ],4,'E',0,'L')  # training data list bottom left 'E' variant 1
    
    trainingDataListE1 =  (71,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0, 
    ],4,'E',0,'L')  # training data list bottom center 'E' variant 1

    trainingDataListE2 =  (72,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1, 
    ],4,'E',0,'L')  # training data list bottom right 'E' variant 1 
    
    trainingDataListE3 =  (73,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],4,'E',0,'L')  # training data list top left 'E' variant 1
    
    trainingDataListE4 =  (74,[
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],4,'E',0,'L')  # training data list top center 'E' variant 1

    trainingDataListE5 =  (75,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],4,'E',0,'L')  # training data list top right 'E' variant 1              
                      
    trainingDataListE6 =  (76,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0, 
    ],4,'E',0,'L')  # training data list bottom left 'E' variant 2
    
    trainingDataListE7 =  (77,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0, 
    ],4,'E',0,'L')  # training data list bottom center 'E' variant 2

    trainingDataListE8 =  (78,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1, 
    ],4,'E',0,'L')  # training data list bottom right 'E' variant 2 
    
    trainingDataListE9 =  (79,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],4,'E',0,'L')  # training data list top left 'E' variant 2
    
    trainingDataListE10 =  (80,[
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],4,'E',0,'L')  # training data list top center 'E' variant 2

    trainingDataListE11 =  (81,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],4,'E',0,'L')  # training data list top right 'E' variant 2                                             
                                        
########
#  F   #
########                        

    trainingDataListF0 =  (6,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom left 'F' variant 1
    
    trainingDataListF1 =  (82,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom center 'F' variant 1

    trainingDataListF2 =  (83,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F') # training data list bottom right 'F' variant 1 
    
    trainingDataListF3 =  (84,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],5,'F',2,'F')  # training data list top left 'F' variant 1
    
    trainingDataListF4 =  (85,[
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list top center 'F' variant 1

    trainingDataListF5 =  (86,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],5,'F',2,'F')  # training data list top right 'F' variant 1              
                      
    trainingDataListF6 =  (87,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom left 'F' variant 2
    
    trainingDataListF7 =  (88,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom center 'F' variant 2

    trainingDataListF8 =  (89,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom right 'F' variant 2 
    
    trainingDataListF9 =  (90,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],5,'F',2,'F')  # training data list top left 'F' variant 2
    
    trainingDataListF10 =  (91,[
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F') # training data list top center 'F' variant 2

    trainingDataListF11 =  (92,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],5,'F',2,'F')  # training data list top right 'F' variant 2                                                     


########
#  G   #
########                        

    trainingDataListG0 =  (7,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0,
    ],6,'G',1,'O')  # training data list bottom left 'G' variant 1
    
    trainingDataListG1 =  (93,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0, 
    ],6,'G',1,'O')  # training data list bottom center 'G' variant 1

    trainingDataListG2 =  (94,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,1,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,  
    ],6,'G',1,'O')  # training data list bottom right 'G' variant 1  
            
    trainingDataListG3 =  (95,[
    0,0,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],6,'G',1,'O')  # training data list top left 'G' variant 1
    
    trainingDataListG4 =  (96,[
    0,0,0,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],6,'G',1,'O')  # training data list top center 'G' variant 1

    trainingDataListG5 =  (97,[
    0,0,0,0,1,1,1,1,1,1,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],6,'G',1,'O')  # training data list top right 'G' variant 1 
    
    trainingDataListG6 =  (98,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],6,'G',1,'O')  # training data list bottom left 'G' variant 2
    
    trainingDataListG7 =  (99,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],6,'G',1,'O')  # training data list bottom center 'G' variant 2

    trainingDataListG8 =  (100,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],6,'G',1,'O')  # training data list bottom right 'G' variant 2  
            
    trainingDataListG9 =  (101,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],6,'G',1,'O')  # training data list top left 'G' variant 2
    
    trainingDataListG10 =  (102,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],6,'G',1,'O')  # training data list top center 'G' variant 2

    trainingDataListG11 =  (103,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],6,'G',1,'O')  # training data list top right 'C' variant 2 
    
########
#  H   #
########     

    trainingDataListH0 =  (8,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],7,'H',4,'N') # training data list bottom left 'H' variant 1
    
    trainingDataListH1 =  (104,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],7,'H',4,'N') # training data list bottom center 'H' variant 1

    trainingDataListH2 =  (105,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],7,'H',4,'N') # training data list bottom right 'H' variant 1  

    trainingDataListH3 =  (106,[
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top left 'H' variant 1
    
    trainingDataListH4 =  (107,[
    0,1,0,0,0,0,0,0,0,1,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top center 'H' variant 1

    trainingDataListH5 =  (108,[ 
    0,0,1,0,0,0,0,0,0,0,1, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],7,'H',4,'N') # training data list top right 'H' variant 1     
    
    trainingDataListH6 =  (109,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    ],7,'H',4,'N') # training data list bottom left 'H' variant 2
    
    trainingDataListH7 =  (110,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    ],7,'H',4,'N') # training data list bottom center 'H' variant 2

    trainingDataListH8 =  (111,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,1,1,1,1,1,1,
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    ],7,'H',4,'N') # training data list bottom right 'H' variant 2  

    trainingDataListH9 =  (112,[
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top left 'H' variant 2
    
    trainingDataListH10 =  (113,[
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top center 'H' variant 2

    trainingDataListH11 =  (114,[
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,1,1,1,1,1,1,
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top right 'H' variant 2   
        
                
########
#  I   #
########     

    trainingDataListI0 =  (9,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    1,1,1,1,1,0,0,0,0,0,0,
    ],8,'I',0,'L') # training data list bottom left 'I' variant 1
    
    trainingDataListI1 =  (115,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    ],8,'I',0,'L') # training data list bottom center 'I' variant 1

    trainingDataListI2 =  (116,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,1,1,1,1,1, 
    ],8,'I',0,'L') # training data list bottom right 'I' variant 1  

    trainingDataListI3 =  (117,[ 
    1,1,1,1,1,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    1,1,1,1,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],8,'I',0,'L') # training data list top left 'I' variant 1
    
    trainingDataListI4 =  (118,[
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],8,'I',0,'L') # training data list top center 'I' variant 1

    trainingDataListI5 =  (119,[
    0,0,0,0,0,0,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],8,'I',0,'L') # training data list bottom right 'I' variant 1  
    
    trainingDataListI6 =  (120,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    ],8,'I',0,'L') # training data list bottom left 'I' variant 2
    
    trainingDataListI7 =  (121,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    ],8,'I',0,'L') # training data list bottom center 'I' variant 2

    trainingDataListI8 =  (122,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1, 
    ],8,'I',0,'L') # training data list bottom right 'I' variant 2 

    trainingDataListI9 =  (123,[
    1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],8,'I',0,'L') # training data list top left 'I' variant 2
    
    trainingDataListI10 =  (124,[ 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],8,'I',0,'L') # training data list top center 'I' variant 2

    trainingDataListI11 =  (125,[ 
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],8,'I',0,'L') # training data list top right 'I' variant 2 

    
########
#  J   #
########     

    trainingDataListJ0 =  (10,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    ],9,'J',1,'O') # training data list bottom left 'J' variant 1
    
    trainingDataListJ1 =  (126,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    ],9,'J',1,'O') # training data list bottom center 'J' variant 1

    trainingDataListJ2 =  (127,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,0,1,1,1,0,0,
    ],9,'J',1,'O') # training data list bottom right 'J' variant 1  
    
    trainingDataListJ3 =  (128,[ 
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],9,'J',1,'O') # training data list top left 'J' variant 1
    
    trainingDataListJ4 =  (129,[ 
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],9,'J',1,'O') # training data list top center 'J' variant 1

    trainingDataListJ5 =  (130,[ 
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,0,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],9,'J',1,'O') # training data list top right 'J' variant 1     
    
    trainingDataListJ6 =  (131,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,0,0,0,0,1,1,0,0,0,
    0,1,1,0,0,1,1,0,0,0,0,
    0,0,1,1,1,1,0,0,0,0,0,
    ],9,'J',1,'O') # training data list bottom left 'J' variant 2
    
    trainingDataListJ7 =  (132,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,0,0,0,0,1,1,0,
    0,0,0,1,1,0,0,1,1,0,0,
    0,0,0,0,1,1,1,1,0,0,0,
    ],9,'J',1,'O') # training data list bottom center 'J' variant 2

    trainingDataListJ8 =  (133,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,1,1,0,0,0,0,1,1,
    0,0,0,0,1,1,0,0,1,1,0,
    0,0,0,0,0,1,1,1,1,0,0,
    ],9,'J',1,'O') # training data list bottom right 'J' variant 2  
    
    trainingDataListJ9 =  (134,[
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,0,0,0,0,1,1,0,0,0,
    0,1,1,0,0,1,1,0,0,0,0,
    0,0,1,1,1,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],9,'J',1,'O') # training data list top left 'J' variant 2
    
    trainingDataListJ10 =  (135,[
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,0,0,0,0,1,1,0,
    0,0,0,1,1,0,0,1,1,0,0,
    0,0,0,0,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],9,'J',1,'O') # training data list top center 'J' variant 2

    trainingDataListJ11 =  (136,[
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,1,1,0,0,0,0,1,1,
    0,0,0,0,1,1,0,0,1,1,0,
    0,0,0,0,0,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],9,'J',1,'O') # training data list topright 'J' variant 2
   
########
#  K   #
########     

    trainingDataListK0 =  (11,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,1,1,0,0,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    ],10,'K',3,'X') # training data list bottom left 'K' variant 1
    
    trainingDataListK1 =  (137,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    ],10,'K',3,'X') # training data list bottom center 'K' variant 1

    trainingDataListK2 =  (138,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,1,0,0,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,1,0,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,1,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,0,0,1,
    ],10,'K',3,'X') # training data list bottom right 'K' variant 1
    
    trainingDataListK3 =  (139,[
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,1,1,0,0,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],10,'K',3,'X') # training data list top left 'K' variant 1
    
    trainingDataListK4 =  (140,[
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top center 'K' variant 1

    trainingDataListK5 =  (141,[
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,1,0,0,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,1,0,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,1,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top right 'K' variant 1
     
    trainingDataListK6 =  (142,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,1,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    1,0,1,0,0,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    ],10,'K',3,'X') # training data list bottom left 'K' variant 2
    
    trainingDataListK7 =  (143,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,1,0,1,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    ],10,'K',3,'X') # training data list bottom center 'K' variant 2

    trainingDataListK8 =  (144,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,1,0,0,0,0,1,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,1,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,1,0,0,1,0,0,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,1,0,0,0,0,1,
    ],10,'K',3,'X') # training data list bottom right 'K' variant 2      
         
    trainingDataListK9 =  (145,[
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,1,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    1,0,1,0,0,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top left 'K' variant 2
    
    trainingDataListK10 =  (146,[
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,1,0,1,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top center 'K' variant 2

    trainingDataListK11 =  (147,[
    0,0,0,0,0,1,0,0,0,0,1,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,1,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,1,0,0,1,0,0,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,1,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top right 'K' variant 2    
    
########
#  L   #
########     

    trainingDataListL0 =  (12,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    ],11,'L',0,'L') # training data list bottom left 'L' variant 1
    
    trainingDataListL1 =  (148,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    ],11,'L',0,'L') # training data list bottom center 'L' variant 1

    trainingDataListL2 =  (149,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    ],11,'L',0,'L') # training data list bottom right 'L' variant 1    
    
    trainingDataListL3 =  (150,[
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],11,'L',0,'L') # training data list top left 'L' variant 1
    
    trainingDataListL4 =  (151,[ 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],11,'L',0,'L') # training data list top center 'L' variant 1

    trainingDataListL5 =  (152,[ 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],11,'L',0,'L') # training data list top right 'L' variant 1  
    
    trainingDataListL6 =  (153,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    ],11,'L',0,'L') # training data list bottom left 'L' variant 2
    
    trainingDataListL7 =  (154,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    ],11,'L',0,'L') # training data list bottom center 'L' variant 2

    trainingDataListL8 =  (155,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,
    ],11,'L',0,'L') # training data list bottom right 'L' variant 2  
    
    trainingDataListL9 =  (156,[ 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],11,'L',0,'L') # training data list top left 'L' variant 2
    
    trainingDataListL10 =  (157,[ 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],11,'L',0,'L') # training data list top center 'L' variant 2

    trainingDataListL11 =  (158,[
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],11,'L',0,'L') # training data list bottom right 'L' variant 2  
                    
########
#  M   #
########     

    trainingDataListM0 =  (13,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,1,0,0,0,1,0,1,0,0,
    1,0,0,1,0,1,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],12,'M',4,'N') # training data list bottom left 'M' variant 1
    
    trainingDataListM1 =  (159,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],12,'M',4,'N') # training data list bottom center 'M' variant 1

    trainingDataListM2 =  (160,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,1,0,0,0,1,0,1,
    0,0,1,0,0,1,0,1,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],12,'M',4,'N') # training data list bottom right 'M' variant 1    
         
    trainingDataListM3 =  (161,[
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,1,0,0,0,1,0,1,0,0,
    1,0,0,1,0,1,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],12,'M',4,'N') # training data list top left 'M' variant 1
    
    trainingDataListM4 =  (162,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],12,'M',4,'N') # training data list top center 'M' variant 1

    trainingDataListM5 =  (163,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,1,0,0,0,1,0,1,
    0,0,1,0,0,1,0,1,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],12,'M',4,'N') # training data list bottom right 'M' variant 1  
   
    trainingDataListM6 =  (164,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,1,1,0,0,0,1,1,1,0,0,
    1,0,1,1,0,1,1,0,1,0,0,
    1,0,0,1,1,1,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],12,'M',4,'N') # training data list bottom left 'M' variant 2
    
    trainingDataListM7 =  (165,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,1,1,0,0,0,1,1,1,0,
    0,1,0,1,1,0,1,1,0,1,0,
    0,1,0,0,1,1,1,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],12,'M',4,'N') # training data list bottom center 'M' variant 2

    trainingDataListM8 =  (166,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,1,1,0,0,0,1,1,1,
    0,0,1,0,1,1,0,1,1,0,1,
    0,0,1,0,0,1,1,1,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],12,'M',4,'N') # training data list bottom right 'M' variant 1         
         
    trainingDataListM9 =  (167,[ 
    0,1,0,0,0,0,0,1,0,0,0,
    1,1,1,0,0,0,1,1,1,0,0,
    1,0,1,1,0,1,1,0,1,0,0,
    1,0,0,1,1,1,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],12,'M',4,'N') # training data list top left 'M' variant 2
    
    trainingDataListM10 =  (168,[
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,1,1,0,0,0,1,1,1,0,
    0,1,0,1,1,0,1,1,0,1,0,
    0,1,0,0,1,1,1,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],12,'M',4,'N') # training data list top center 'M' variant 2

    trainingDataListM11 =  (169,[ 
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,1,1,0,0,0,1,1,1,
    0,0,1,0,1,1,0,1,1,0,1,
    0,0,1,0,0,1,1,1,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],12,'M',4,'N') # training data list top right 'M' variant 1               
                  

########
#  N   #
########     

    trainingDataListN0 =  (14,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,0,1,0,0,
    1,0,1,0,0,0,0,0,1,0,0,
    1,0,0,1,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],13,'N',4,'N') # training data list bottom left 'N' variant 1
    
    trainingDataListN1 =  (170,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,0,1,0,
    0,1,0,1,0,0,0,0,0,1,0,
    0,1,0,0,1,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],13,'N',4,'N') # training data list bottom center 'N' variant 1

    trainingDataListN2 =  (171,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,0,1,
    0,0,1,0,1,0,0,0,0,0,1,
    0,0,1,0,0,1,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],13,'N',4,'N') # training data list bottom right 'N' variant 1    

    trainingDataListN3 =  (172,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,0,1,0,0,
    1,0,1,0,0,0,0,0,1,0,0,
    1,0,0,1,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],13,'N',4,'N') # training data list top left 'N' variant 1
    
    trainingDataListN4 =  (173,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,0,1,0,
    0,1,0,1,0,0,0,0,0,1,0,
    0,1,0,0,1,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],13,'N',4,'N') # training data list top center 'N' variant 1

    trainingDataListN5 =  (174,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,0,1,
    0,0,1,0,1,0,0,0,0,0,1,
    0,0,1,0,0,1,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],13,'N',4,'N') # training data list top right 'N' variant 1   

    trainingDataListN6 =  (175,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,0,0,0,0,0,1,0,0,0,
    1,1,1,0,0,0,0,1,0,0,0,
    1,0,1,1,0,0,0,1,0,0,0,
    1,0,0,1,1,0,0,1,0,0,0,
    1,0,0,0,1,1,0,1,0,0,0,
    1,0,0,0,0,1,1,1,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    ],13,'N',4,'N') # training data list bottom left 'N' variant 2
    
    trainingDataListN7 =  (176,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,0,0,0,0,0,1,0,0,
    0,1,1,1,0,0,0,0,1,0,0,
    0,1,0,1,1,0,0,0,1,0,0,
    0,1,0,0,1,1,0,0,1,0,0,
    0,1,0,0,0,1,1,0,1,0,0,
    0,1,0,0,0,0,1,1,1,0,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    ],13,'N',4,'N') # training data list bottom center 'N' variant 2

    trainingDataListN8 =  (177,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,1,1,0,0,0,0,0,1,
    0,0,0,1,1,1,0,0,0,0,1,
    0,0,0,1,0,1,1,0,0,0,1,
    0,0,0,1,0,0,1,1,0,0,1,
    0,0,0,1,0,0,0,1,1,0,1,
    0,0,0,1,0,0,0,0,1,1,1,
    0,0,0,1,0,0,0,0,0,1,1,
    0,0,0,1,0,0,0,0,0,0,1,
    ],13,'N',4,'N') # training data list bottom right 'N' variant 2    

    trainingDataListN9 =  (178,[
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,0,0,0,0,0,1,0,0,0,
    1,1,1,0,0,0,0,1,0,0,0,
    1,0,1,1,0,0,0,1,0,0,0,
    1,0,0,1,1,0,0,1,0,0,0,
    1,0,0,0,1,1,0,1,0,0,0,
    1,0,0,0,0,1,1,1,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],13,'N',4,'N') # training data list top left 'N' variant 2
    
    trainingDataListN10 =  (179,[ 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,0,0,0,0,0,1,0,0,
    0,1,1,1,0,0,0,0,1,0,0,
    0,1,0,1,1,0,0,0,1,0,0,
    0,1,0,0,1,1,0,0,1,0,0,
    0,1,0,0,0,1,1,0,1,0,0,
    0,1,0,0,0,0,1,1,1,0,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],13,'N',4,'N') # training data list top center 'N' variant 2

    trainingDataListN11 =  (180,[ 
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,1,1,0,0,0,0,0,1,
    0,0,0,1,1,1,0,0,0,0,1,
    0,0,0,1,0,1,1,0,0,0,1,
    0,0,0,1,0,0,1,1,0,0,1,
    0,0,0,1,0,0,0,1,1,0,1,
    0,0,0,1,0,0,0,0,1,1,1,
    0,0,0,1,0,0,0,0,0,1,1,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],13,'N',4,'N') # training data list top right 'N' variant 1    

########
#  O   #
########                        

    trainingDataListO0 =  (15,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0, 
    ],14,'O',1,'O')  # training data list bottom left 'O' variant 1
    
    trainingDataListO1 =  (181,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    ],14,'O',1,'O')  # training data list bottom center 'O' variant 1

    trainingDataListO2 =  (182,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0, 
    ],14,'O',1,'O')  # training data list bottom right 'O' variant 1  
    
    trainingDataListO3 =  (183,[
    0,0,1,1,1,1,1,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],14,'O',1,'O')  # training data list top left 'O' variant 1
    
    trainingDataListO4 =  (184,[ 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],14,'O',1,'O')  # training data list top center 'O' variant 1

    trainingDataListO5 =  (185,[
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],14,'O',1,'O')  # training data list top right 'O' varian

    trainingDataListO6 =  (186,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],14,'O',1,'O')  # training data list bottom left 'O' variant 2
    
    trainingDataListO7 =  (187,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],14,'O',1,'O')  # training data list bottom center 'O' variant 2

    trainingDataListO8 =  (188,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],14,'O',1,'O')  # training data list bottom right 'O' variant 2  
    
    trainingDataListO9 =  (189,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],14,'O',1,'O')  # training data list top left 'O' variant 2
    
    trainingDataListO10 =  (190,[ 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],14,'O',1,'O')  # training data list top center 'O' variant 2

    trainingDataListO11 =  (191,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],14,'O',1,'O')  # training data list top right 'O' variant 2  

########
#  P   #
########                        

    trainingDataListP0 =  (16,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F')  # training data list bottom left 'P' variant 1
    
    trainingDataListP1 =  (192,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F')  # training data list bottom center 'P' variant 1

    trainingDataListP2 =  (193,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F') # training data list bottom right 'P' variant 1 
    
    trainingDataListP3 =  (194,[ 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],15,'P',2,'F')  # training data list top left 'P' variant 1
    
    trainingDataListP4 =  (195,[
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],15,'P',2,'F')  # training data list top center 'P' variant 1

    trainingDataListP5 =  (196,[ 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F') # training data list top right 'P' variant 1 
                      
    trainingDataListP6 =  (197,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F')  # training data list bottom left 'P' variant 2
    
    trainingDataListP7 =  (198,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F')  # training data list bottom center 'P' variant 2

    trainingDataListP8 =  (199,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F') # training data list bottom right 'P' variant 2 
    
    trainingDataListP9 =  (200,[ 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],15,'P',2,'F')  # training data list top left 'P' variant 2
    
    trainingDataListP10 =  (201,[
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],15,'P',2,'F')  # training data list top center 'P' variant 2

    trainingDataListP11 =  (202,[ 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F') # training data list top right 'P' variant 2 
 
########
#  Q   #
########                        

    trainingDataListQ0 =  (17,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,1,0,0, 
    ],16,'Q',1,'O') # training data list bottom left 'Q' variant 1
    
    trainingDataListQ1 =  (203,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,1,0,  
    ],16,'Q',1,'O')  # training data list bottom center 'Q' variant 1

    trainingDataListQ2 =  (204,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,1, 
    ],16,'Q',1,'O')  # training data list bottom right 'Q' variant 1  
    
    trainingDataListQ3 =  (205,[
    0,0,1,1,1,1,1,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],16,'Q',1,'O')  # training data list top left 'Q' variant 1
    
    trainingDataListQ4 =  (206,[ 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,1,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],16,'Q',1,'O')  # training data list top center 'Q' variant 1

    trainingDataListQ5 =  (207,[
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],16,'Q',1,'O')  # training data list top right 'Q' variant 1  
    
    trainingDataListQ6 =  (208,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],16,'Q',1,'O') # training data list bottom left 'Q' variant 2
    
    trainingDataListQ7 =  (209,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],16,'Q',1,'O')  # training data list bottom center 'Q' variant 2

    trainingDataListQ8 =  (210,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],16,'Q',1,'O')  # training data list bottom right 'Q' variant 2  
    
    trainingDataListQ9 =  (211,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],16,'Q',1,'O')  # training data list top left 'Q' variant 2
    
    trainingDataListQ10 =  (212,[ 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],16,'Q',1,'O')  # training data list top center 'Q' variant 2

    trainingDataListQ11 =  (213,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],16,'Q',1,'O') # training data list top right 'Q' variant 2             
                               
########
#  R   #
########                        

    trainingDataListR0 =  (18,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0, 
    ],17,'R',2,'F')  # training data list bottom left 'R' variant 1
    
    trainingDataListR1 =  (214,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0, 
    ],17,'R',2,'F')  # training data list bottom center 'R' variant 1

    trainingDataListR2 =  (215,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1, 
    ],17,'R',2,'F') # training data list bottom right 'R' variant 1 
    
    trainingDataListR3 =  (216,[ 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],17,'R',2,'F')  # training data list top left 'R' variant 1
    
    trainingDataListR4 =  (217,[
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],17,'R',2,'F')  # training data list top center 'R' variant 1

    trainingDataListR5 =  (218,[ 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],17,'R',2,'F') # training data list top right 'R' variant 1 
                      
    trainingDataListR6 =  (219,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0, 
    ],17,'R',2,'F')  # training data list bottom left 'R' variant 2
    
    trainingDataListR7 =  (220,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0, 
    ],17,'R',2,'F')  # training data list bottom center 'R' variant 2

    trainingDataListR8 =  (221,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1, 
    ],17,'R',2,'F')  # training data list bottom right 'R' variant 2 
    
    trainingDataListR9 =  (222,[ 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],17,'R',2,'F')  # training data list top left 'R' variant 2
    
    trainingDataListR10 =  (223,[
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],17,'R',2,'F')  # training data list top center 'R' variant 2

    trainingDataListR11 =  (224,[ 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],17,'R',2,'F') # training data list top right 'R' variant 2   

########
#  S  #
########                        

    trainingDataListS0 =  (19,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],18,'S',1,'O')  # training data list bottom left 'S' variant 1
    
    trainingDataListS1 =  (225,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    ],18,'S',1,'O')  # training data list bottom center 'S' variant 1

    trainingDataListS2 =  (226,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,1,1,1,1,1,1,0, 
    ],18,'S',1,'O') # training data list bottom right 'S' variant 1 
    
    trainingDataListS3 =  (227,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],18,'S',1,'O')  # training data list top left 'S' variant 1
    
    trainingDataListS4 =  (228,[ 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],18,'S',1,'O')  # training data list top center 'S' variant 1

    trainingDataListS5 =  (229,[ 
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],18,'S',1,'O') # training data list top right 'S' variant 1 
    
    trainingDataListS6 =  (230,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],18,'S',1,'O')  # training data list bottom left 'S' variant 2
    
    trainingDataListS7 =  (231,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,
    0,0,0,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    ],18,'S',1,'O')  # training data list bottom center 'S' variant 2

    trainingDataListS8 =  (232,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,1,1,1,0,
    0,0,0,0,0,0,0,0,0,1,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,
    ],18,'S',1,'O') # training data list bottom right 'S' variant 2 
    
    trainingDataListS9 =  (233,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],18,'S',1,'O')  # training data list top left 'S' variant 2
    
    trainingDataListS10 =  (234,[  
    0,0,1,1,1,1,1,1,1,0,0, 
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,
    0,0,0,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],18,'S',1,'O')  # training data list top center 'S' variant 2

    trainingDataListS11 =  (235,[ 
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,1,1,1,0,
    0,0,0,0,0,0,0,0,0,1,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],18,'S',1,'O') # training data list top right 'S' variant 2 

########
#  T  #
########                        

    trainingDataListT0 =  (20,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0, 
    ],19,'T',2,'F')  # training data list bottom left 'T' variant 1
    
    trainingDataListT1 =  (236,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list bottom center 'T' variant 1

    trainingDataListT2 =  (237,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],19,'T',2,'F') # training data list bottom right 'T' variant 1   
    
    trainingDataListT3 =  (238,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list top left 'T' variant 1
    
    trainingDataListT4 =  (239,[ 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list top center 'T' variant 1

    trainingDataListT5 =  (240,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],19,'T',2,'F') # training data list top right 'T' variant 1
    
    trainingDataListT6 =  (241,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list bottom left 'T' variant 2
    
    trainingDataListT7 =  (242,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list bottom center 'T' variant 2

    trainingDataListT8 =  (243,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    ],19,'T',2,'F') # training data list bottom right 'T' variant 2   
    
    trainingDataListT9 =  (244,[ 
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],19,'T',2,'F')  # training data list top left 'T' variant 2
    
    trainingDataListT10 =  (245,[  
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list top center 'T' variant 2

    trainingDataListT11 =  (246,[ 
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],19,'T',2,'F') # training data list top right 'T' variant 2


########
#  U  #
########                        

    trainingDataListU0 =  (21,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0,
    ],20,'U',4,'N')  # training data list bottom left 'U' variant 1
    
    trainingDataListU1 =  (247,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    ],20,'U',4,'N')  # training data list bottom center 'U' variant 1

    trainingDataListU2 =  (248,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,
    ],20,'U',4,'N') # training data list bottom right 'U' variant 1      
    
    trainingDataListU3 =  (249,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N')  # training data list top left 'U' variant 1
    
    trainingDataListU4 =  (250,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N')  # training data list top center 'U' variant 1

    trainingDataListU5 =  (251,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N') # training data list top right 'U' variant 1 
    
    trainingDataListU6 =  (252,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    ],20,'U',4,'N')  # training data list bottom left 'U' variant 2
    
    trainingDataListU7 =  (253,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,
    ],20,'U',4,'N')  # training data list bottom center 'U' variant 2

    trainingDataListU8 =  (254,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,
    ],20,'U',4,'N') # training data list bottom right 'U' variant 2      
    
    trainingDataListU9 =  (255,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N')  # training data list top left 'U' variant 2
    
    trainingDataListU10 =  (256,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N')  # training data list top center 'U' variant 2

    trainingDataListU11 =  (257,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N') # training data list top right 'U' variant 2 
    
########
#  V  #
########                        

    trainingDataListV0 =  (22,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list bottom left 'V' variant 1
    
    trainingDataListV1 =  (258,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list bottom center 'V' variant 1

    trainingDataListV2 =  (259,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],21,'V',3,'X') # training data list bottom right 'V' variant 1   
    
    trainingDataListV3 =  (260,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list top left 'V' variant 1
    
    trainingDataListV4 =  (261,[
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list top center 'V' variant 1

    trainingDataListV5 =  (262,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X') # training data list top right 'V' variant 1     
    
    trainingDataListV6 =  (263,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list bottom left 'V' variant 2
    
    trainingDataListV7 =  (264,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list bottom center 'V' variant 2

    trainingDataListV8 =  (265,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],21,'V',3,'X') # training data list bottom right 'V' variant 2   
    
    trainingDataListV9 =  (266,[  
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],21,'V',3,'X')  # training data list top left 'V' variant 2
    
    trainingDataListV10 =  (267,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list top center 'V' variant 2

    trainingDataListV11 =  (268,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X') # training data list top right 'V' variant 2   
    
    
########
#  W  #
########                        

    trainingDataListW0 =  (23,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,1,0,1,0,0,1,0,0,
    1,0,1,0,0,0,1,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],22,'W',4,'N')  # training data list bottom left 'W' variant 1
    
    trainingDataListW1 =  (269,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],22,'W',4,'N')  # training data list bottom center 'W' variant 1

    trainingDataListW2 =  (270,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,1,
    0,0,1,0,1,0,0,0,1,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],22,'W',4,'N') # training data list bottom right 'W' variant 1      
    
    trainingDataListW3 =  (271,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,1,0,1,0,0,1,0,0,
    1,0,1,0,0,0,1,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N')  # training data list top left 'W' variant 1
    
    trainingDataListW4 =  (272,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N')  # training data list top center 'W' variant 1

    trainingDataListW5 =  (273,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,1,
    0,0,1,0,1,0,0,0,1,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N') # training data list top right 'W' variant 1  


    trainingDataListW6 =  (274,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,1,1,1,0,0,1,0,0,
    1,0,1,1,0,1,1,0,1,0,0,
    1,1,1,0,0,0,1,1,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    ],22,'W',4,'N')  # training data list bottom left 'W' variant 2
    
    trainingDataListW7 =  (275,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,1,1,1,0,0,1,0,
    0,1,0,1,1,0,1,1,0,1,0,
    0,1,1,1,0,0,0,1,1,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    ],22,'W',4,'N')  # training data list bottom center 'W' variant 2

    trainingDataListW8 =  (276,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,1,1,1,0,0,1,
    0,0,1,0,1,1,0,1,1,0,1,
    0,0,1,1,1,0,0,0,1,1,1,
    0,0,0,1,0,0,0,0,0,1,0,
    ],22,'W',4,'N') # training data list bottom right 'W' variant 2      
    
    trainingDataListW9 =  (277,[  
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,1,1,1,0,0,1,0,0,
    1,0,1,1,0,1,1,0,1,0,0,
    1,1,1,0,0,0,1,1,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N')  # training data list top left 'W' variant 2
    
    trainingDataListW10 =  (278,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N')  # training data list top center 'W' variant 2

    trainingDataListW11 =  (279,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,1,1,1,0,0,1,
    0,0,1,0,1,1,0,1,1,0,1,
    0,0,1,1,1,0,0,0,1,1,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],22,'W',4,'N') # training data list top right 'W' variant 2  

########
#  X  #
########                        

    trainingDataListX0 =  (24,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],23,'X',3,'X')  # training data list bottom left 'X' variant 1
    
    trainingDataListX1 =  (280,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],23,'X',3,'X')  # training data list bottom center 'X' variant 1

    trainingDataListX2 =  (281,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    ],23,'X',3,'X') # training data list bottom right 'X' variant 1  
    
    trainingDataListX3 =  (282,[
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],23,'X',3,'X')  # training data list top left 'X' variant 1
    
    trainingDataListX4 =  (283,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],23,'X',3,'X')  # training data list top center 'X' variant 1

    trainingDataListX5 =  (284,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],23,'X',3,'X') # training data list top right 'X' variant 1  
    
    trainingDataListX6 =  (285,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],23,'X',3,'X')  # training data list bottom left 'X' variant 2
    
    trainingDataListX7 =  (286,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],23,'X',3,'X')  # training data list bottom center 'X' variant 2

    trainingDataListX8 =  (287,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],23,'X',3,'X') # training data list bottom right 'X' variant 2 
    
    trainingDataListX9 =  (288,[
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],23,'X',3,'X')  # training data list top left 'X' variant 2
    
    trainingDataListX10 =  (289,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],23,'X',3,'X')  # training data list top center 'X' variant 2

    trainingDataListX11 =  (290,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],23,'X',3,'X') # training data list top right 'X' variant 2  
    
########
#  Y  #
########                        

    trainingDataListY0 =  (25,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    ],24,'Y',3,'X')   # training data list bottom left 'Y' variant 1
    
    trainingDataListY1 =  (291,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],24,'Y',3,'X')  # training data list bottom center 'Y' variant 1

    trainingDataListY2 =  (292,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],24,'Y',3,'X') # training data list bottom right 'Y' variant 1  
    
    trainingDataListY3 =  (293,[
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],24,'Y',3,'X')  # training data list top left 'Y' variant 1
    
    trainingDataListY4 =  (294,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],24,'Y',3,'X')  # training data list top center 'Y variant 1

    trainingDataListY5 =  (295,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],24,'Y',3,'X') # training data list top right 'Y' variant 1  
    
    trainingDataListY6 =  (296,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    ],24,'Y',3,'X')  # training data list bottom left 'Y' variant 2
    
    trainingDataListY7 =  (297,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],24,'Y',3,'X')  # training data list bottom center 'Y' variant 2

    trainingDataListY8 =  (298,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],24,'Y',3,'X') # training data list bottom right 'Y' variant 2 
    
    trainingDataListY9 =  (299,[
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],24,'Y',3,'X')  # training data list top left 'Y' variant 2
    
    trainingDataListY10 =  (300,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],24,'Y',3,'X')  # training data list top center 'Y' variant 2

    trainingDataListY11 =  (301,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],24,'Y',3,'X') # training data list top right 'Y' variant 2
  
########
#  Z  #
########                        

    trainingDataListZ0 =  (26,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    ],25,'Z',0,'L')    # training data list bottom left 'Z' variant 1
    
    trainingDataListZ1 =  (302,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    ],25,'Z',0,'L')   # training data list bottom center 'Z' variant 1

    trainingDataListZ2 =  (303,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    ],25,'Z',0,'L')  # training data list bottom right 'Z' variant 1  
    
    trainingDataListZ3 =  (304,[ 
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],25,'Z',0,'L')    # training data list top left 'Z' variant 1
    
    trainingDataListZ4 =  (305,[
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0
    ],25,'Z',0,'L')   # training data list top center 'Z' variant 1

    trainingDataListZ5 =  (306,[
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],25,'Z',0,'L')  # training data list top right 'Z' variant 1  
    
    trainingDataListZ6 =  (307,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    ],25,'Z',0,'L')    # training data list bottom left 'Z' variant 2
    
    trainingDataListZ7 =  (308,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    ],25,'Z',0,'L')   # training data list bottom center 'Z' variant 2

    trainingDataListZ8 =  (309,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    ],25,'Z',0,'L')  # training data list bottom right 'Z' variant 2  
    
    trainingDataListZ9 =  (310,[ 
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],25,'Z',0,'L')    # training data list top left 'Z' variant 2
    
    trainingDataListZ10 =  (311,[
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0
    ],25,'Z',0,'L')   # training data list top center 'Z' variant 2

    trainingDataListZ11 =  (312,[
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],25,'Z',0,'L')  # training data list top right 'Z' variant 2 

       
    if dataSet == 1: trainingDataList = trainingDataListA0
    if dataSet == 2: trainingDataList = trainingDataListA1    
    if dataSet == 3: trainingDataList = trainingDataListA2
    if dataSet == 4: trainingDataList = trainingDataListA3
    if dataSet == 5: trainingDataList = trainingDataListA4        
    if dataSet == 6: trainingDataList = trainingDataListA5
    if dataSet == 7: trainingDataList = trainingDataListA6
    if dataSet == 8: trainingDataList = trainingDataListA7   
    if dataSet == 9: trainingDataList = trainingDataListA8 
    if dataSet == 10: trainingDataList = trainingDataListA9
    if dataSet == 11: trainingDataList = trainingDataListA10  
    if dataSet == 12: trainingDataList = trainingDataListA11 
                                                                                                 
    if dataSet == 13: trainingDataList = trainingDataListB0
    if dataSet == 14: trainingDataList = trainingDataListB1    
    if dataSet == 15: trainingDataList = trainingDataListB2
    if dataSet == 16: trainingDataList = trainingDataListB3
    if dataSet == 17: trainingDataList = trainingDataListB4        
    if dataSet == 18: trainingDataList = trainingDataListB5
    if dataSet == 19: trainingDataList = trainingDataListB6
    if dataSet == 20: trainingDataList = trainingDataListB7   
    if dataSet == 21: trainingDataList = trainingDataListB8 
    if dataSet == 22: trainingDataList = trainingDataListB9
    if dataSet == 23: trainingDataList = trainingDataListB10  
    if dataSet == 24: trainingDataList = trainingDataListB11 
    
    if dataSet == 25: trainingDataList = trainingDataListC0
    if dataSet == 26: trainingDataList = trainingDataListC1    
    if dataSet == 27: trainingDataList = trainingDataListC2
    if dataSet == 28: trainingDataList = trainingDataListC3
    if dataSet == 29: trainingDataList = trainingDataListC4        
    if dataSet == 30: trainingDataList = trainingDataListC5
    if dataSet == 31: trainingDataList = trainingDataListC6
    if dataSet == 32: trainingDataList = trainingDataListC7   
    if dataSet == 33: trainingDataList = trainingDataListC8 
    if dataSet == 34: trainingDataList = trainingDataListC9
    if dataSet == 35: trainingDataList = trainingDataListC10  
    if dataSet == 36: trainingDataList = trainingDataListC11 
    
    if dataSet == 37: trainingDataList = trainingDataListD0
    if dataSet == 38: trainingDataList = trainingDataListD1    
    if dataSet == 39: trainingDataList = trainingDataListD2
    if dataSet == 40: trainingDataList = trainingDataListD3
    if dataSet == 41: trainingDataList = trainingDataListD4        
    if dataSet == 42: trainingDataList = trainingDataListD5
    if dataSet == 43: trainingDataList = trainingDataListD6
    if dataSet == 44: trainingDataList = trainingDataListD7   
    if dataSet == 45: trainingDataList = trainingDataListD8 
    if dataSet == 46: trainingDataList = trainingDataListD9
    if dataSet == 47: trainingDataList = trainingDataListD10  
    if dataSet == 48: trainingDataList = trainingDataListD11 
    
    if dataSet == 49: trainingDataList = trainingDataListE0
    if dataSet == 50: trainingDataList = trainingDataListE1    
    if dataSet == 51: trainingDataList = trainingDataListE2
    if dataSet == 52: trainingDataList = trainingDataListE3
    if dataSet == 53: trainingDataList = trainingDataListE4        
    if dataSet == 54: trainingDataList = trainingDataListE5
    if dataSet == 55: trainingDataList = trainingDataListE6
    if dataSet == 56: trainingDataList = trainingDataListE7   
    if dataSet == 57: trainingDataList = trainingDataListE8 
    if dataSet == 58: trainingDataList = trainingDataListE9
    if dataSet == 59: trainingDataList = trainingDataListE10  
    if dataSet == 60: trainingDataList = trainingDataListE11 
    
    if dataSet == 61: trainingDataList = trainingDataListF0
    if dataSet == 62: trainingDataList = trainingDataListF1    
    if dataSet == 63: trainingDataList = trainingDataListF2
    if dataSet == 64: trainingDataList = trainingDataListF3
    if dataSet == 65: trainingDataList = trainingDataListF4        
    if dataSet == 66: trainingDataList = trainingDataListF5
    if dataSet == 67: trainingDataList = trainingDataListF6
    if dataSet == 68: trainingDataList = trainingDataListF7   
    if dataSet == 69: trainingDataList = trainingDataListF8 
    if dataSet == 70: trainingDataList = trainingDataListF9
    if dataSet == 71: trainingDataList = trainingDataListF10  
    if dataSet == 72: trainingDataList = trainingDataListF11 
    
    if dataSet == 73: trainingDataList = trainingDataListG0
    if dataSet == 74: trainingDataList = trainingDataListG1    
    if dataSet == 75: trainingDataList = trainingDataListG2
    if dataSet == 76: trainingDataList = trainingDataListG3
    if dataSet == 77: trainingDataList = trainingDataListG4        
    if dataSet == 78: trainingDataList = trainingDataListG5
    if dataSet == 79: trainingDataList = trainingDataListG6
    if dataSet == 80: trainingDataList = trainingDataListG7   
    if dataSet == 81: trainingDataList = trainingDataListG8 
    if dataSet == 82: trainingDataList = trainingDataListG9
    if dataSet == 83: trainingDataList = trainingDataListG10  
    if dataSet == 84: trainingDataList = trainingDataListG11     
    
    if dataSet == 85: trainingDataList = trainingDataListH0
    if dataSet == 86: trainingDataList = trainingDataListH1    
    if dataSet == 87: trainingDataList = trainingDataListH2
    if dataSet == 88: trainingDataList = trainingDataListH3
    if dataSet == 89: trainingDataList = trainingDataListH4        
    if dataSet == 90: trainingDataList = trainingDataListH5
    if dataSet == 91: trainingDataList = trainingDataListH6
    if dataSet == 92: trainingDataList = trainingDataListH7   
    if dataSet == 93: trainingDataList = trainingDataListH8 
    if dataSet == 94: trainingDataList = trainingDataListH9
    if dataSet == 95: trainingDataList = trainingDataListH10  
    if dataSet == 96: trainingDataList = trainingDataListH11                                                                                                                                                                                                                
    
    if dataSet == 97: trainingDataList = trainingDataListI0
    if dataSet == 98: trainingDataList = trainingDataListI1    
    if dataSet == 99: trainingDataList = trainingDataListI2
    if dataSet == 100: trainingDataList = trainingDataListI3
    if dataSet == 101: trainingDataList = trainingDataListI4        
    if dataSet == 102: trainingDataList = trainingDataListI5
    if dataSet == 103: trainingDataList = trainingDataListI6
    if dataSet == 104: trainingDataList = trainingDataListI7   
    if dataSet == 105: trainingDataList = trainingDataListI8 
    if dataSet == 106: trainingDataList = trainingDataListI9
    if dataSet == 107: trainingDataList = trainingDataListI10  
    if dataSet == 108: trainingDataList = trainingDataListI11       
    
    if dataSet == 109: trainingDataList = trainingDataListJ0
    if dataSet == 110: trainingDataList = trainingDataListJ1    
    if dataSet == 111: trainingDataList = trainingDataListJ2
    if dataSet == 112: trainingDataList = trainingDataListJ3
    if dataSet == 113: trainingDataList = trainingDataListJ4        
    if dataSet == 114: trainingDataList = trainingDataListJ5
    if dataSet == 115: trainingDataList = trainingDataListJ6
    if dataSet == 116: trainingDataList = trainingDataListJ7   
    if dataSet == 117: trainingDataList = trainingDataListJ8 
    if dataSet == 118: trainingDataList = trainingDataListJ9
    if dataSet == 119: trainingDataList = trainingDataListJ10  
    if dataSet == 120: trainingDataList = trainingDataListJ11   
    
    if dataSet == 121: trainingDataList = trainingDataListK0
    if dataSet == 122: trainingDataList = trainingDataListK1    
    if dataSet == 123: trainingDataList = trainingDataListK2
    if dataSet == 124: trainingDataList = trainingDataListK3
    if dataSet == 125: trainingDataList = trainingDataListK4        
    if dataSet == 126: trainingDataList = trainingDataListK5
    if dataSet == 127: trainingDataList = trainingDataListK6
    if dataSet == 128: trainingDataList = trainingDataListK7   
    if dataSet == 129: trainingDataList = trainingDataListK8 
    if dataSet == 130: trainingDataList = trainingDataListK9
    if dataSet == 131: trainingDataList = trainingDataListK10  
    if dataSet == 132: trainingDataList = trainingDataListK11     
    
    if dataSet == 133: trainingDataList = trainingDataListL0
    if dataSet == 134: trainingDataList = trainingDataListL1    
    if dataSet == 135: trainingDataList = trainingDataListL2
    if dataSet == 136: trainingDataList = trainingDataListL3
    if dataSet == 137: trainingDataList = trainingDataListL4        
    if dataSet == 138: trainingDataList = trainingDataListL5
    if dataSet == 139: trainingDataList = trainingDataListL6
    if dataSet == 140: trainingDataList = trainingDataListL7   
    if dataSet == 141: trainingDataList = trainingDataListL8 
    if dataSet == 142: trainingDataList = trainingDataListL9
    if dataSet == 143: trainingDataList = trainingDataListL10  
    if dataSet == 144: trainingDataList = trainingDataListL11     
    
    if dataSet == 145: trainingDataList = trainingDataListM0
    if dataSet == 146: trainingDataList = trainingDataListM1    
    if dataSet == 147: trainingDataList = trainingDataListM2
    if dataSet == 148: trainingDataList = trainingDataListM3
    if dataSet == 149: trainingDataList = trainingDataListM4        
    if dataSet == 150: trainingDataList = trainingDataListM5
    if dataSet == 151: trainingDataList = trainingDataListM6
    if dataSet == 152: trainingDataList = trainingDataListM7   
    if dataSet == 153: trainingDataList = trainingDataListM8 
    if dataSet == 154: trainingDataList = trainingDataListM9
    if dataSet == 155: trainingDataList = trainingDataListM10  
    if dataSet == 156: trainingDataList = trainingDataListM11    
    
    if dataSet == 157: trainingDataList = trainingDataListN0
    if dataSet == 158: trainingDataList = trainingDataListN1    
    if dataSet == 159: trainingDataList = trainingDataListN2
    if dataSet == 160: trainingDataList = trainingDataListN3
    if dataSet == 161: trainingDataList = trainingDataListN4        
    if dataSet == 162: trainingDataList = trainingDataListN5
    if dataSet == 163: trainingDataList = trainingDataListN6
    if dataSet == 164: trainingDataList = trainingDataListN7   
    if dataSet == 165: trainingDataList = trainingDataListN8 
    if dataSet == 166: trainingDataList = trainingDataListN9
    if dataSet == 167: trainingDataList = trainingDataListN10  
    if dataSet == 168: trainingDataList = trainingDataListN11     
    
    if dataSet == 169: trainingDataList = trainingDataListO0
    if dataSet == 170: trainingDataList = trainingDataListO1    
    if dataSet == 171: trainingDataList = trainingDataListO2
    if dataSet == 172: trainingDataList = trainingDataListO3
    if dataSet == 173: trainingDataList = trainingDataListO4        
    if dataSet == 174: trainingDataList = trainingDataListO5
    if dataSet == 175: trainingDataList = trainingDataListO6
    if dataSet == 176: trainingDataList = trainingDataListO7   
    if dataSet == 177: trainingDataList = trainingDataListO8 
    if dataSet == 178: trainingDataList = trainingDataListO9
    if dataSet == 179: trainingDataList = trainingDataListO10  
    if dataSet == 180: trainingDataList = trainingDataListO11    
    
    if dataSet == 181: trainingDataList = trainingDataListP0
    if dataSet == 182: trainingDataList = trainingDataListP1    
    if dataSet == 183: trainingDataList = trainingDataListP2
    if dataSet == 184: trainingDataList = trainingDataListP3
    if dataSet == 185: trainingDataList = trainingDataListP4        
    if dataSet == 186: trainingDataList = trainingDataListP5
    if dataSet == 187: trainingDataList = trainingDataListP6
    if dataSet == 188: trainingDataList = trainingDataListP7   
    if dataSet == 189: trainingDataList = trainingDataListP8 
    if dataSet == 190: trainingDataList = trainingDataListP9
    if dataSet == 191: trainingDataList = trainingDataListP10  
    if dataSet == 192: trainingDataList = trainingDataListP11    
    
    if dataSet == 193: trainingDataList = trainingDataListQ0
    if dataSet == 194: trainingDataList = trainingDataListQ1    
    if dataSet == 195: trainingDataList = trainingDataListQ2
    if dataSet == 196: trainingDataList = trainingDataListQ3
    if dataSet == 197: trainingDataList = trainingDataListQ4        
    if dataSet == 198: trainingDataList = trainingDataListQ5
    if dataSet == 199: trainingDataList = trainingDataListQ6
    if dataSet == 200: trainingDataList = trainingDataListQ7   
    if dataSet == 201: trainingDataList = trainingDataListQ8 
    if dataSet == 202: trainingDataList = trainingDataListQ9
    if dataSet == 203: trainingDataList = trainingDataListQ10  
    if dataSet == 204: trainingDataList = trainingDataListQ11  
    
    if dataSet == 205: trainingDataList = trainingDataListR0
    if dataSet == 206: trainingDataList = trainingDataListR1    
    if dataSet == 207: trainingDataList = trainingDataListR2
    if dataSet == 208: trainingDataList = trainingDataListR3
    if dataSet == 209: trainingDataList = trainingDataListR4        
    if dataSet == 210: trainingDataList = trainingDataListR5
    if dataSet == 211: trainingDataList = trainingDataListR6
    if dataSet == 212: trainingDataList = trainingDataListR7   
    if dataSet == 213: trainingDataList = trainingDataListR8 
    if dataSet == 214: trainingDataList = trainingDataListR9
    if dataSet == 215: trainingDataList = trainingDataListR10  
    if dataSet == 216: trainingDataList = trainingDataListR11 
    
    if dataSet == 217: trainingDataList = trainingDataListS0
    if dataSet == 218: trainingDataList = trainingDataListS1    
    if dataSet == 219: trainingDataList = trainingDataListS2
    if dataSet == 220: trainingDataList = trainingDataListS3
    if dataSet == 221: trainingDataList = trainingDataListS4        
    if dataSet == 222: trainingDataList = trainingDataListS5
    if dataSet == 223: trainingDataList = trainingDataListS6
    if dataSet == 224: trainingDataList = trainingDataListS7   
    if dataSet == 225: trainingDataList = trainingDataListS8 
    if dataSet == 226: trainingDataList = trainingDataListS9
    if dataSet == 227: trainingDataList = trainingDataListS10  
    if dataSet == 228: trainingDataList = trainingDataListS11     
    
    if dataSet == 229: trainingDataList = trainingDataListT0
    if dataSet == 230: trainingDataList = trainingDataListT1    
    if dataSet == 231: trainingDataList = trainingDataListT2
    if dataSet == 232: trainingDataList = trainingDataListT3
    if dataSet == 233: trainingDataList = trainingDataListT4        
    if dataSet == 234: trainingDataList = trainingDataListT5
    if dataSet == 235: trainingDataList = trainingDataListT6
    if dataSet == 236: trainingDataList = trainingDataListT7   
    if dataSet == 237: trainingDataList = trainingDataListT8 
    if dataSet == 238: trainingDataList = trainingDataListT9
    if dataSet == 239: trainingDataList = trainingDataListT10  
    if dataSet == 240: trainingDataList = trainingDataListT11        
    
    if dataSet == 241: trainingDataList = trainingDataListU0
    if dataSet == 242: trainingDataList = trainingDataListU1    
    if dataSet == 243: trainingDataList = trainingDataListU2
    if dataSet == 244: trainingDataList = trainingDataListU3
    if dataSet == 245: trainingDataList = trainingDataListU4        
    if dataSet == 246: trainingDataList = trainingDataListU5
    if dataSet == 247: trainingDataList = trainingDataListU6
    if dataSet == 248: trainingDataList = trainingDataListU7   
    if dataSet == 249: trainingDataList = trainingDataListU8 
    if dataSet == 250: trainingDataList = trainingDataListU9
    if dataSet == 251: trainingDataList = trainingDataListU10  
    if dataSet == 252: trainingDataList = trainingDataListU11            
    
    if dataSet == 253: trainingDataList = trainingDataListV0
    if dataSet == 254: trainingDataList = trainingDataListV1    
    if dataSet == 255: trainingDataList = trainingDataListV2
    if dataSet == 256: trainingDataList = trainingDataListV3
    if dataSet == 257: trainingDataList = trainingDataListV4        
    if dataSet == 258: trainingDataList = trainingDataListV5
    if dataSet == 259: trainingDataList = trainingDataListV6
    if dataSet == 260: trainingDataList = trainingDataListV7   
    if dataSet == 261: trainingDataList = trainingDataListV8 
    if dataSet == 262: trainingDataList = trainingDataListV9
    if dataSet == 263: trainingDataList = trainingDataListV10  
    if dataSet == 264: trainingDataList = trainingDataListV11                 
    
    if dataSet == 265: trainingDataList = trainingDataListW0
    if dataSet == 266: trainingDataList = trainingDataListW1    
    if dataSet == 267: trainingDataList = trainingDataListW2
    if dataSet == 268: trainingDataList = trainingDataListW3
    if dataSet == 269: trainingDataList = trainingDataListW4        
    if dataSet == 270: trainingDataList = trainingDataListW5
    if dataSet == 271: trainingDataList = trainingDataListW6
    if dataSet == 272: trainingDataList = trainingDataListW7   
    if dataSet == 273: trainingDataList = trainingDataListW8 
    if dataSet == 274: trainingDataList = trainingDataListW9
    if dataSet == 275: trainingDataList = trainingDataListW10  
    if dataSet == 276: trainingDataList = trainingDataListW11 
    
    if dataSet == 277: trainingDataList = trainingDataListX0
    if dataSet == 278: trainingDataList = trainingDataListX1    
    if dataSet == 279: trainingDataList = trainingDataListX2
    if dataSet == 280: trainingDataList = trainingDataListX3
    if dataSet == 281: trainingDataList = trainingDataListX4        
    if dataSet == 282: trainingDataList = trainingDataListX5
    if dataSet == 283: trainingDataList = trainingDataListX6
    if dataSet == 284: trainingDataList = trainingDataListX7   
    if dataSet == 285: trainingDataList = trainingDataListX8 
    if dataSet == 286: trainingDataList = trainingDataListX9
    if dataSet == 287: trainingDataList = trainingDataListX10  
    if dataSet == 288: trainingDataList = trainingDataListX11 
    
    if dataSet == 289: trainingDataList = trainingDataListY0
    if dataSet == 290: trainingDataList = trainingDataListY1    
    if dataSet == 291: trainingDataList = trainingDataListY2
    if dataSet == 292: trainingDataList = trainingDataListY3
    if dataSet == 293: trainingDataList = trainingDataListY4        
    if dataSet == 294: trainingDataList = trainingDataListY5
    if dataSet == 295: trainingDataList = trainingDataListY6
    if dataSet == 296: trainingDataList = trainingDataListY7   
    if dataSet == 297: trainingDataList = trainingDataListY8 
    if dataSet == 298: trainingDataList = trainingDataListY9
    if dataSet == 299: trainingDataList = trainingDataListY10  
    if dataSet == 300: trainingDataList = trainingDataListY11 
    
    if dataSet == 301: trainingDataList = trainingDataListZ0
    if dataSet == 302: trainingDataList = trainingDataListZ1    
    if dataSet == 303: trainingDataList = trainingDataListZ2
    if dataSet == 304: trainingDataList = trainingDataListZ3
    if dataSet == 305: trainingDataList = trainingDataListZ4        
    if dataSet == 306: trainingDataList = trainingDataListZ5
    if dataSet == 307: trainingDataList = trainingDataListZ6
    if dataSet == 308: trainingDataList = trainingDataListZ7   
    if dataSet == 309: trainingDataList = trainingDataListZ8 
    if dataSet == 310: trainingDataList = trainingDataListZ9
    if dataSet == 311: trainingDataList = trainingDataListZ10  
    if dataSet == 312: trainingDataList = trainingDataListZ11               
                                 
    return (trainingDataList)      

   
####################################################################################################
####################################################################################################
#
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
#
####################################################################################################
####################################################################################################

def obtainRandomAlphabetTrainingValues (numTrainingDataSets):
      
    dataSet = random.randint(0, numTrainingDataSets)

########
#  A   #
########

    trainingDataListA0 =  (1,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],0,'A',2,'F') # training data list bottom left 'A' variant 1
    
    trainingDataListA1 =  (27,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,1,0,0,0,0,0, 
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],0,'A',2,'F') # training data list bottom center 'A' variant 1

    trainingDataListA2 =  (28,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,1,0,0,0,0, 
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],0,'A',2,'F') # training data list bottom right 'A' variant 1  
    

    trainingDataListA3 =  (29,[
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    ],0,'A',2,'F') # training data list top left 'A' variant 1
    
    trainingDataListA4 =  (30,[
    0,0,0,0,0,1,0,0,0,0,0, 
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],0,'A',2,'F') # training data list top center 'A' variant 1

    trainingDataListA5 =  (31,[
    0,0,0,0,0,0,1,0,0,0,0, 
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],0,'A',2,'F') # training data list top right 'A' variant 1    
    
           
    trainingDataListA6 =  (32,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],0,'A',2,'F') # training data list bottom left 'A' variant 2
    
    trainingDataListA7 =  (33,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,1,0,0,0,0,0, 
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,    
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],0,'A',2,'F') # training data list bottom center 'A' variant 2

    trainingDataListA8 =  (34,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,1,0,0,0,0, 
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,    
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],0,'A',2,'F') # training data list bottom right 'A' variant 2 
    

    trainingDataListA9 =  (35,[
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,   
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    ],0,'A',2,'F') # training data list top left 'A' variant 2
    
    trainingDataListA10 =  (36,[
    0,0,0,0,0,1,0,0,0,0,0, 
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,    
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],0,'A',2,'F') # training data list top center 'A' variant 2

    trainingDataListA11 =  (37,[
    0,0,0,0,0,0,1,0,0,0,0, 
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,    
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],0,'A',2,'F') # training data list top right 'A' variant 2          
              

########
#  B   #
########                        

    trainingDataListB0 =  (2,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    ],1,'B',0,'L') # training data list bottom left 'B' variant 1
    
    trainingDataListB1 =  (38,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],1,'B',0,'L') # training data list bottom center 'B' variant 1

    trainingDataListB2 =  (39,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],1,'B',0,'L') # training data list bottom right 'B' variant 1  
            
    trainingDataListB3 =  (40,[
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],1,'B',0,'L') # training data list top left 'B' variant 1
    
    trainingDataListB4 =  (41,[
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],1,'B',0,'L') # training data list top center 'B' variant 1

    trainingDataListB5 =  (42,[
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],1,'B',0,'L') # training data list top right 'B' variant 1  
    
    trainingDataListB6 =  (43,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0, 
    ],1,'B',0,'L') # training data list bottom left 'B' variant 2
    
    trainingDataListB7 =  (44,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0, 
    ],1,'B',0,'L') # training data list bottom center 'B' variant 2

    trainingDataListB8 =  (45,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0, 
    ],1,'B',0,'L') # training data list bottom right 'B' variant 2    
    
    trainingDataListB9 =  (46,[
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],1,'B',0,'L') # training data list top left 'B' variant 2
    
    trainingDataListB10 =  (47,[
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],1,'B',0,'L') # training data list top center 'B' variant 2

    trainingDataListB11 =  (48,[
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],1,'B',0,'L') # training data list top right 'B' variant 2   
    
    

########
#  C   #
########                        

    trainingDataListC0 =  (3,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],2,'C',1,'O')  # training data list bottom left 'C' variant 1
    
    trainingDataListC1 =  (49,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],2,'C',1,'O')  # training data list bottom center 'C' variant 1

    trainingDataListC2 =  (50,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],2,'C',1,'O')  # training data list bottom right 'C' variant 1  
            
    trainingDataListC3 =  (51,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top left 'C' variant 1
    
    trainingDataListC4 =  (52,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],2,'C',1,'O')  # training data list top center 'C' variant 1

    trainingDataListC5 =  (53,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top right 'C' variant 1  
    
    trainingDataListC6 =  (54,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],2,'C',1,'O')  # training data list bottom left 'C' variant 2
    
    trainingDataListC7 =  (55,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],2,'C',1,'O')  # training data list bottom center 'C' variant 2

    trainingDataListC8 =  (56,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],2,'C',1,'O')  # training data list bottom right 'C' variant 2  
            
    trainingDataListC9 =  (57,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top left 'C' variant 2
    
    trainingDataListC10 =  (58,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],2,'C',1,'O')  # training data list top center 'C' variant 2

    trainingDataListC11 =  (59,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top right 'C' variant 2 ########
#  B   #
########                        

    trainingDataListB0 =  (2,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    ],1,'B',0,'L') # training data list bottom left 'B' variant 1
    
    trainingDataListB1 =  (38,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],1,'B',0,'L') # training data list bottom center 'B' variant 1

    trainingDataListB2 =  (39,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],1,'B',0,'L') # training data list bottom right 'B' variant 1  
            
    trainingDataListB3 =  (40,[
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],1,'B',0,'L') # training data list top left 'B' variant 1
    
    trainingDataListB4 =  (41,[
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],1,'B',0,'L') # training data list top center 'B' variant 1

    trainingDataListB5 =  (42,[
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],1,'B',0,'L') # training data list top right 'B' variant 1  
    
    trainingDataListB6 =  (43,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0, 
    ],1,'B',0,'L') # training data list bottom left 'B' variant 2
    
    trainingDataListB7 =  (44,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0, 
    ],1,'B',0,'L') # training data list bottom center 'B' variant 2

    trainingDataListB8 =  (45,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0, 
    ],1,'B',0,'L') # training data list bottom right 'B' variant 2    
    
    trainingDataListB9 =  (46,[
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],1,'B',0,'L') # training data list top left 'B' variant 2
    
    trainingDataListB10 =  (47,[
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],1,'B',0,'L') # training data list top center 'B' variant 2

    trainingDataListB11 =  (48,[
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],1,'B',0,'L') # training data list top right 'B' variant 2   
    
    

########
#  C   #
########                        

    trainingDataListC0 =  (3,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,0,1,1,1,1,1,1,0,0,0, 
    ],2,'C',1,'O')  # training data list bottom left 'C' variant 1
    
    trainingDataListC1 =  (49,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,0,1,1,1,1,1,1,0,0,  
    ],2,'C',1,'O')  # training data list bottom center 'C' variant 1

    trainingDataListC2 =  (50,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,1,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,0,1,1,1,1,1,1,0,
    ],2,'C',1,'O')  # training data list bottom right 'C' variant 1  
            
    trainingDataListC3 =  (51,[
    0,0,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,0,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top left 'C' variant 1
    
    trainingDataListC4 =  (52,[
    0,0,0,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,0,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],2,'C',1,'O')  # training data list top center 'C' variant 1

    trainingDataListC5 =  (53,[
    0,0,0,0,1,1,1,1,1,1,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,0,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top right 'C' variant 1  
    
    trainingDataListC6 =  (54,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],2,'C',1,'O')  # training data list bottom left 'C' variant 2
    
    trainingDataListC7 =  (55,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],2,'C',1,'O')  # training data list bottom center 'C' variant 2

    trainingDataListC8 =  (56,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],2,'C',1,'O')  # training data list bottom right 'C' variant 2  
            
    trainingDataListC9 =  (57,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top left 'C' variant 2
    
    trainingDataListC10 =  (58,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],2,'C',1,'O')  # training data list top center 'C' variant 2

    trainingDataListC11 =  (59,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],2,'C',1,'O')  # training data list top right 'C' variant 2 
    
    
########
#  D   #
########                        

    trainingDataListD0 =  (4,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    ],3,'D',1,'O')  # training data list bottom left 'D' variant 1
    
    trainingDataListD1 =  (60,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],3,'D',1,'O')  # training data list bottom center 'D' variant 1

    trainingDataListD2 =  (61,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],3,'D',1,'O')  # training data list bottom right 'D' variant 1      
    
    trainingDataListD3 =  (62,[
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],3,'D',1,'O')  # training data list top left 'D' variant 1
    
    trainingDataListD4 =  (63,[ 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],3,'D',1,'O')  # training data list top center 'D' variant 1

    trainingDataListD5 =  (64,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],3,'D',1,'O')  # training data list top right 'D' variant 1                     
    
    trainingDataListD6 =  (65,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,1,1,0,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    ],3,'D',1,'O')  # training data list bottom left 'D' variant 2
    
    trainingDataListD7 =  (66,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],3,'D',1,'O')  # training data list bottom center 'D' variant 2

    trainingDataListD8 =  (67,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],3,'D',1,'O')  # training data list bottom right 'D' variant 2     
    
    trainingDataListD9 =  (68,[
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,1,1,0,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,1,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],3,'D',1,'O')  # training data list top left 'D' variant 2
    
    trainingDataListD10 =  (69,[ 
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],3,'D',1,'O')  # training data list top center 'D' variant 2

    trainingDataListD11 =  (70,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,1,0,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],3,'D',1,'O')  # training data list top right 'D' variant 2                     


########
#  E   #
########                        

    trainingDataListE0 =  (5,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0, 
    ],4,'E',0,'L')  # training data list bottom left 'E' variant 1
    
    trainingDataListE1 =  (71,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0, 
    ],4,'E',0,'L')  # training data list bottom center 'E' variant 1

    trainingDataListE2 =  (72,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1, 
    ],4,'E',0,'L')  # training data list bottom right 'E' variant 1 
    
    trainingDataListE3 =  (73,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],4,'E',0,'L')  # training data list top left 'E' variant 1
    
    trainingDataListE4 =  (74,[
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],4,'E',0,'L')  # training data list top center 'E' variant 1

    trainingDataListE5 =  (75,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],4,'E',0,'L')  # training data list top right 'E' variant 1              
                      
    trainingDataListE6 =  (76,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0, 
    ],4,'E',0,'L')  # training data list bottom left 'E' variant 2
    
    trainingDataListE7 =  (77,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0, 
    ],4,'E',0,'L')  # training data list bottom center 'E' variant 2

    trainingDataListE8 =  (78,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1, 
    ],4,'E',0,'L')  # training data list bottom right 'E' variant 2 
    
    trainingDataListE9 =  (79,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],4,'E',0,'L')  # training data list top left 'E' variant 2
    
    trainingDataListE10 =  (80,[
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],4,'E',0,'L')  # training data list top center 'E' variant 2

    trainingDataListE11 =  (81,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],4,'E',0,'L')  # training data list top right 'E' variant 2                                             
                                        
########
#  F   #
########                        

    trainingDataListF0 =  (6,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom left 'F' variant 1
    
    trainingDataListF1 =  (82,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom center 'F' variant 1

    trainingDataListF2 =  (83,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F') # training data list bottom right 'F' variant 1 
    
    trainingDataListF3 =  (84,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],5,'F',2,'F')  # training data list top left 'F' variant 1
    
    trainingDataListF4 =  (85,[
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list top center 'F' variant 1

    trainingDataListF5 =  (86,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],5,'F',2,'F')  # training data list top right 'F' variant 1              
                      
    trainingDataListF6 =  (87,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom left 'F' variant 2
    
    trainingDataListF7 =  (88,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom center 'F' variant 2

    trainingDataListF8 =  (89,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F')  # training data list bottom right 'F' variant 2 
    
    trainingDataListF9 =  (90,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],5,'F',2,'F')  # training data list top left 'F' variant 2
    
    trainingDataListF10 =  (91,[
    0,1,1,1,1,1,1,1,1,1,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],5,'F',2,'F') # training data list top center 'F' variant 2

    trainingDataListF11 =  (92,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],5,'F',2,'F')  # training data list top right 'F' variant 2                                                     


########
#  G   #
########                        

    trainingDataListG0 =  (7,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0,
    ],6,'G',1,'O')  # training data list bottom left 'G' variant 1
    
    trainingDataListG1 =  (93,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0, 
    ],6,'G',1,'O')  # training data list bottom center 'G' variant 1

    trainingDataListG2 =  (94,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,1,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,  
    ],6,'G',1,'O')  # training data list bottom right 'G' variant 1  
            
    trainingDataListG3 =  (95,[
    0,0,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],6,'G',1,'O')  # training data list top left 'G' variant 1
    
    trainingDataListG4 =  (96,[
    0,0,0,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],6,'G',1,'O')  # training data list top center 'G' variant 1

    trainingDataListG5 =  (97,[
    0,0,0,0,1,1,1,1,1,1,0,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],6,'G',1,'O')  # training data list top right 'G' variant 1 
    
    trainingDataListG6 =  (98,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],6,'G',1,'O')  # training data list bottom left 'G' variant 2
    
    trainingDataListG7 =  (99,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],6,'G',1,'O')  # training data list bottom center 'G' variant 2

    trainingDataListG8 =  (100,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],6,'G',1,'O')  # training data list bottom right 'G' variant 2  
            
    trainingDataListG9 =  (101,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],6,'G',1,'O')  # training data list top left 'G' variant 2
    
    trainingDataListG10 =  (102,[
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],6,'G',1,'O')  # training data list top center 'G' variant 2

    trainingDataListG11 =  (103,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],6,'G',1,'O')  # training data list top right 'C' variant 2 
    
########
#  H   #
########     

    trainingDataListH0 =  (8,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],7,'H',4,'N') # training data list bottom left 'H' variant 1
    
    trainingDataListH1 =  (104,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],7,'H',4,'N') # training data list bottom center 'H' variant 1

    trainingDataListH2 =  (105,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],7,'H',4,'N') # training data list bottom right 'H' variant 1  

    trainingDataListH3 =  (106,[
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top left 'H' variant 1
    
    trainingDataListH4 =  (107,[
    0,1,0,0,0,0,0,0,0,1,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top center 'H' variant 1

    trainingDataListH5 =  (108,[ 
    0,0,1,0,0,0,0,0,0,0,1, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],7,'H',4,'N') # training data list top right 'H' variant 1     
    
    trainingDataListH6 =  (109,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    ],7,'H',4,'N') # training data list bottom left 'H' variant 2
    
    trainingDataListH7 =  (110,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    ],7,'H',4,'N') # training data list bottom center 'H' variant 2

    trainingDataListH8 =  (111,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,1,1,1,1,1,1,
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    ],7,'H',4,'N') # training data list bottom right 'H' variant 2  

    trainingDataListH9 =  (112,[
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top left 'H' variant 2
    
    trainingDataListH10 =  (113,[
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,1,0,0,0,0,0,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top center 'H' variant 2

    trainingDataListH11 =  (114,[
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,1,1,1,1,1,1,
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,1,0,0,0,0,0,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],7,'H',4,'N') # training data list top right 'H' variant 2   
        
                
########
#  I   #
########     

    trainingDataListI0 =  (9,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    1,1,1,1,1,0,0,0,0,0,0,
    ],8,'I',0,'L') # training data list bottom left 'I' variant 1
    
    trainingDataListI1 =  (115,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    ],8,'I',0,'L') # training data list bottom center 'I' variant 1

    trainingDataListI2 =  (116,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,1,1,1,1,1, 
    ],8,'I',0,'L') # training data list bottom right 'I' variant 1  

    trainingDataListI3 =  (117,[ 
    1,1,1,1,1,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    1,1,1,1,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],8,'I',0,'L') # training data list top left 'I' variant 1
    
    trainingDataListI4 =  (118,[
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],8,'I',0,'L') # training data list top center 'I' variant 1

    trainingDataListI5 =  (119,[
    0,0,0,0,0,0,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],8,'I',0,'L') # training data list bottom right 'I' variant 1  
    
    trainingDataListI6 =  (120,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    ],8,'I',0,'L') # training data list bottom left 'I' variant 2
    
    trainingDataListI7 =  (121,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    ],8,'I',0,'L') # training data list bottom center 'I' variant 2

    trainingDataListI8 =  (122,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1, 
    ],8,'I',0,'L') # training data list bottom right 'I' variant 2 

    trainingDataListI9 =  (123,[
    1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],8,'I',0,'L') # training data list top left 'I' variant 2
    
    trainingDataListI10 =  (124,[ 
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],8,'I',0,'L') # training data list top center 'I' variant 2

    trainingDataListI11 =  (125,[ 
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],8,'I',0,'L') # training data list top right 'I' variant 2 

    
########
#  J   #
########     

    trainingDataListJ0 =  (10,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    ],9,'J',1,'O') # training data list bottom left 'J' variant 1
    
    trainingDataListJ1 =  (126,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    ],9,'J',1,'O') # training data list bottom center 'J' variant 1

    trainingDataListJ2 =  (127,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,0,1,1,1,0,0,
    ],9,'J',1,'O') # training data list bottom right 'J' variant 1  
    
    trainingDataListJ3 =  (128,[ 
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],9,'J',1,'O') # training data list top left 'J' variant 1
    
    trainingDataListJ4 =  (129,[ 
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],9,'J',1,'O') # training data list top center 'J' variant 1

    trainingDataListJ5 =  (130,[ 
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,0,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],9,'J',1,'O') # training data list top right 'J' variant 1     
    
    trainingDataListJ6 =  (131,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,0,0,0,0,1,1,0,0,0,
    0,1,1,0,0,1,1,0,0,0,0,
    0,0,1,1,1,1,0,0,0,0,0,
    ],9,'J',1,'O') # training data list bottom left 'J' variant 2
    
    trainingDataListJ7 =  (132,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,0,0,0,0,1,1,0,
    0,0,0,1,1,0,0,1,1,0,0,
    0,0,0,0,1,1,1,1,0,0,0,
    ],9,'J',1,'O') # training data list bottom center 'J' variant 2

    trainingDataListJ8 =  (133,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,1,1,0,0,0,0,1,1,
    0,0,0,0,1,1,0,0,1,1,0,
    0,0,0,0,0,1,1,1,1,0,0,
    ],9,'J',1,'O') # training data list bottom right 'J' variant 2  
    
    trainingDataListJ9 =  (134,[
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,0,0,0,0,1,1,0,0,0,
    0,1,1,0,0,1,1,0,0,0,0,
    0,0,1,1,1,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],9,'J',1,'O') # training data list top left 'J' variant 2
    
    trainingDataListJ10 =  (135,[
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,0,0,0,0,1,1,0,
    0,0,0,1,1,0,0,1,1,0,0,
    0,0,0,0,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],9,'J',1,'O') # training data list top center 'J' variant 2

    trainingDataListJ11 =  (136,[
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,1,1,0,0,0,0,1,1,
    0,0,0,0,1,1,0,0,1,1,0,
    0,0,0,0,0,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],9,'J',1,'O') # training data list topright 'J' variant 2
   
########
#  K   #
########     

    trainingDataListK0 =  (11,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,1,1,0,0,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    ],10,'K',3,'X') # training data list bottom left 'K' variant 1
    
    trainingDataListK1 =  (137,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    ],10,'K',3,'X') # training data list bottom center 'K' variant 1

    trainingDataListK2 =  (138,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,1,0,0,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,1,0,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,1,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,0,0,1,
    ],10,'K',3,'X') # training data list bottom right 'K' variant 1
    
    trainingDataListK3 =  (139,[
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,1,1,0,0,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],10,'K',3,'X') # training data list top left 'K' variant 1
    
    trainingDataListK4 =  (140,[
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top center 'K' variant 1

    trainingDataListK5 =  (141,[
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,1,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,1,0,0,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,1,0,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,1,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top right 'K' variant 1
     
    trainingDataListK6 =  (142,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,1,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    1,0,1,0,0,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    ],10,'K',3,'X') # training data list bottom left 'K' variant 2
    
    trainingDataListK7 =  (143,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,1,0,1,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    ],10,'K',3,'X') # training data list bottom center 'K' variant 2

    trainingDataListK8 =  (144,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,1,0,0,0,0,1,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,1,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,1,0,0,1,0,0,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,1,0,0,0,0,1,
    ],10,'K',3,'X') # training data list bottom right 'K' variant 2      
         
    trainingDataListK9 =  (145,[
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,1,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    1,0,1,0,0,0,0,0,0,0,0,
    1,0,0,1,0,0,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top left 'K' variant 2
    
    trainingDataListK10 =  (146,[
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,1,0,1,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top center 'K' variant 2

    trainingDataListK11 =  (147,[
    0,0,0,0,0,1,0,0,0,0,1,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,1,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,1,0,0,1,0,0,
    0,0,0,0,0,1,0,0,0,1,0,
    0,0,0,0,0,1,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],10,'K',3,'X') # training data list top right 'K' variant 2    
    
########
#  L   #
########     

    trainingDataListL0 =  (12,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    ],11,'L',0,'L') # training data list bottom left 'L' variant 1
    
    trainingDataListL1 =  (148,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    ],11,'L',0,'L') # training data list bottom center 'L' variant 1

    trainingDataListL2 =  (149,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    ],11,'L',0,'L') # training data list bottom right 'L' variant 1    
    
    trainingDataListL3 =  (150,[
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],11,'L',0,'L') # training data list top left 'L' variant 1
    
    trainingDataListL4 =  (151,[ 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],11,'L',0,'L') # training data list top center 'L' variant 1

    trainingDataListL5 =  (152,[ 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],11,'L',0,'L') # training data list top right 'L' variant 1  
    
    trainingDataListL6 =  (153,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    ],11,'L',0,'L') # training data list bottom left 'L' variant 2
    
    trainingDataListL7 =  (154,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    ],11,'L',0,'L') # training data list bottom center 'L' variant 2

    trainingDataListL8 =  (155,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,
    ],11,'L',0,'L') # training data list bottom right 'L' variant 2  
    
    trainingDataListL9 =  (156,[ 
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],11,'L',0,'L') # training data list top left 'L' variant 2
    
    trainingDataListL10 =  (157,[ 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],11,'L',0,'L') # training data list top center 'L' variant 2

    trainingDataListL11 =  (158,[
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],11,'L',0,'L') # training data list bottom right 'L' variant 2  
                    
########
#  M   #
########     

    trainingDataListM0 =  (13,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,1,0,0,0,1,0,1,0,0,
    1,0,0,1,0,1,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],12,'M',4,'N') # training data list bottom left 'M' variant 1
    
    trainingDataListM1 =  (159,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],12,'M',4,'N') # training data list bottom center 'M' variant 1

    trainingDataListM2 =  (160,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,1,0,0,0,1,0,1,
    0,0,1,0,0,1,0,1,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],12,'M',4,'N') # training data list bottom right 'M' variant 1    
         
    trainingDataListM3 =  (161,[
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,1,0,0,0,1,0,1,0,0,
    1,0,0,1,0,1,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],12,'M',4,'N') # training data list top left 'M' variant 1
    
    trainingDataListM4 =  (162,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],12,'M',4,'N') # training data list top center 'M' variant 1

    trainingDataListM5 =  (163,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,1,0,0,0,1,0,1,
    0,0,1,0,0,1,0,1,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],12,'M',4,'N') # training data list bottom right 'M' variant 1  
   
    trainingDataListM6 =  (164,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,1,1,0,0,0,1,1,1,0,0,
    1,0,1,1,0,1,1,0,1,0,0,
    1,0,0,1,1,1,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],12,'M',4,'N') # training data list bottom left 'M' variant 2
    
    trainingDataListM7 =  (165,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,1,1,0,0,0,1,1,1,0,
    0,1,0,1,1,0,1,1,0,1,0,
    0,1,0,0,1,1,1,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],12,'M',4,'N') # training data list bottom center 'M' variant 2

    trainingDataListM8 =  (166,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,1,1,0,0,0,1,1,1,
    0,0,1,0,1,1,0,1,1,0,1,
    0,0,1,0,0,1,1,1,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],12,'M',4,'N') # training data list bottom right 'M' variant 1         
         
    trainingDataListM9 =  (167,[ 
    0,1,0,0,0,0,0,1,0,0,0,
    1,1,1,0,0,0,1,1,1,0,0,
    1,0,1,1,0,1,1,0,1,0,0,
    1,0,0,1,1,1,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],12,'M',4,'N') # training data list top left 'M' variant 2
    
    trainingDataListM10 =  (168,[
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,1,1,0,0,0,1,1,1,0,
    0,1,0,1,1,0,1,1,0,1,0,
    0,1,0,0,1,1,1,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],12,'M',4,'N') # training data list top center 'M' variant 2

    trainingDataListM11 =  (169,[ 
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,1,1,0,0,0,1,1,1,
    0,0,1,0,1,1,0,1,1,0,1,
    0,0,1,0,0,1,1,1,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],12,'M',4,'N') # training data list top right 'M' variant 1               
                  

########
#  N   #
########     

    trainingDataListN0 =  (14,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,0,1,0,0,
    1,0,1,0,0,0,0,0,1,0,0,
    1,0,0,1,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],13,'N',4,'N') # training data list bottom left 'N' variant 1
    
    trainingDataListN1 =  (170,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,0,1,0,
    0,1,0,1,0,0,0,0,0,1,0,
    0,1,0,0,1,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],13,'N',4,'N') # training data list bottom center 'N' variant 1

    trainingDataListN2 =  (171,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,0,1,
    0,0,1,0,1,0,0,0,0,0,1,
    0,0,1,0,0,1,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],13,'N',4,'N') # training data list bottom right 'N' variant 1    

    trainingDataListN3 =  (172,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,0,1,0,0,
    1,0,1,0,0,0,0,0,1,0,0,
    1,0,0,1,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],13,'N',4,'N') # training data list top left 'N' variant 1
    
    trainingDataListN4 =  (173,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,0,1,0,
    0,1,0,1,0,0,0,0,0,1,0,
    0,1,0,0,1,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],13,'N',4,'N') # training data list top center 'N' variant 1

    trainingDataListN5 =  (174,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,0,1,
    0,0,1,0,1,0,0,0,0,0,1,
    0,0,1,0,0,1,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],13,'N',4,'N') # training data list top right 'N' variant 1   

    trainingDataListN6 =  (175,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,0,0,0,0,0,1,0,0,0,
    1,1,1,0,0,0,0,1,0,0,0,
    1,0,1,1,0,0,0,1,0,0,0,
    1,0,0,1,1,0,0,1,0,0,0,
    1,0,0,0,1,1,0,1,0,0,0,
    1,0,0,0,0,1,1,1,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    ],13,'N',4,'N') # training data list bottom left 'N' variant 2
    
    trainingDataListN7 =  (176,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,0,0,0,0,0,1,0,0,
    0,1,1,1,0,0,0,0,1,0,0,
    0,1,0,1,1,0,0,0,1,0,0,
    0,1,0,0,1,1,0,0,1,0,0,
    0,1,0,0,0,1,1,0,1,0,0,
    0,1,0,0,0,0,1,1,1,0,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    ],13,'N',4,'N') # training data list bottom center 'N' variant 2

    trainingDataListN8 =  (177,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,1,1,0,0,0,0,0,1,
    0,0,0,1,1,1,0,0,0,0,1,
    0,0,0,1,0,1,1,0,0,0,1,
    0,0,0,1,0,0,1,1,0,0,1,
    0,0,0,1,0,0,0,1,1,0,1,
    0,0,0,1,0,0,0,0,1,1,1,
    0,0,0,1,0,0,0,0,0,1,1,
    0,0,0,1,0,0,0,0,0,0,1,
    ],13,'N',4,'N') # training data list bottom right 'N' variant 2    

    trainingDataListN9 =  (178,[
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,0,0,0,0,0,1,0,0,0,
    1,1,1,0,0,0,0,1,0,0,0,
    1,0,1,1,0,0,0,1,0,0,0,
    1,0,0,1,1,0,0,1,0,0,0,
    1,0,0,0,1,1,0,1,0,0,0,
    1,0,0,0,0,1,1,1,0,0,0,
    1,0,0,0,0,0,1,1,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],13,'N',4,'N') # training data list top left 'N' variant 2
    
    trainingDataListN10 =  (179,[ 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,0,0,0,0,0,1,0,0,
    0,1,1,1,0,0,0,0,1,0,0,
    0,1,0,1,1,0,0,0,1,0,0,
    0,1,0,0,1,1,0,0,1,0,0,
    0,1,0,0,0,1,1,0,1,0,0,
    0,1,0,0,0,0,1,1,1,0,0,
    0,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],13,'N',4,'N') # training data list top center 'N' variant 2

    trainingDataListN11 =  (180,[ 
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,1,1,0,0,0,0,0,1,
    0,0,0,1,1,1,0,0,0,0,1,
    0,0,0,1,0,1,1,0,0,0,1,
    0,0,0,1,0,0,1,1,0,0,1,
    0,0,0,1,0,0,0,1,1,0,1,
    0,0,0,1,0,0,0,0,1,1,1,
    0,0,0,1,0,0,0,0,0,1,1,
    0,0,0,1,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],13,'N',4,'N') # training data list top right 'N' variant 1    

########
#  O   #
########                        

    trainingDataListO0 =  (15,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0, 
    ],14,'O',1,'O')  # training data list bottom left 'O' variant 1
    
    trainingDataListO1 =  (181,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    ],14,'O',1,'O')  # training data list bottom center 'O' variant 1

    trainingDataListO2 =  (182,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0, 
    ],14,'O',1,'O')  # training data list bottom right 'O' variant 1  
    
    trainingDataListO3 =  (183,[
    0,0,1,1,1,1,1,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],14,'O',1,'O')  # training data list top left 'O' variant 1
    
    trainingDataListO4 =  (184,[ 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],14,'O',1,'O')  # training data list top center 'O' variant 1

    trainingDataListO5 =  (185,[
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],14,'O',1,'O')  # training data list top right 'O' varian

    trainingDataListO6 =  (186,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],14,'O',1,'O')  # training data list bottom left 'O' variant 2
    
    trainingDataListO7 =  (187,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],14,'O',1,'O')  # training data list bottom center 'O' variant 2

    trainingDataListO8 =  (188,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],14,'O',1,'O')  # training data list bottom right 'O' variant 2  
    
    trainingDataListO9 =  (189,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],14,'O',1,'O')  # training data list top left 'O' variant 2
    
    trainingDataListO10 =  (190,[ 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],14,'O',1,'O')  # training data list top center 'O' variant 2

    trainingDataListO11 =  (191,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],14,'O',1,'O')  # training data list top right 'O' variant 2  

########
#  P   #
########                        

    trainingDataListP0 =  (16,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F')  # training data list bottom left 'P' variant 1
    
    trainingDataListP1 =  (192,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F')  # training data list bottom center 'P' variant 1

    trainingDataListP2 =  (193,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F') # training data list bottom right 'P' variant 1 
    
    trainingDataListP3 =  (194,[ 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],15,'P',2,'F')  # training data list top left 'P' variant 1
    
    trainingDataListP4 =  (195,[
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],15,'P',2,'F')  # training data list top center 'P' variant 1

    trainingDataListP5 =  (196,[ 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F') # training data list top right 'P' variant 1 
                      
    trainingDataListP6 =  (197,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F')  # training data list bottom left 'P' variant 2
    
    trainingDataListP7 =  (198,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F')  # training data list bottom center 'P' variant 2

    trainingDataListP8 =  (199,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F') # training data list bottom right 'P' variant 2 
    
    trainingDataListP9 =  (200,[ 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],15,'P',2,'F')  # training data list top left 'P' variant 2
    
    trainingDataListP10 =  (201,[
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],15,'P',2,'F')  # training data list top center 'P' variant 2

    trainingDataListP11 =  (202,[ 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],15,'P',2,'F') # training data list top right 'P' variant 2 
 
########
#  Q   #
########                        

    trainingDataListQ0 =  (17,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,1,0,0, 
    ],16,'Q',1,'O') # training data list bottom left 'Q' variant 1
    
    trainingDataListQ1 =  (203,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,1,0,  
    ],16,'Q',1,'O')  # training data list bottom center 'Q' variant 1

    trainingDataListQ2 =  (204,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,1, 
    ],16,'Q',1,'O')  # training data list bottom right 'Q' variant 1  
    
    trainingDataListQ3 =  (205,[
    0,0,1,1,1,1,1,0,0,0,0, 
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],16,'Q',1,'O')  # training data list top left 'Q' variant 1
    
    trainingDataListQ4 =  (206,[ 
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,1,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],16,'Q',1,'O')  # training data list top center 'Q' variant 1

    trainingDataListQ5 =  (207,[
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],16,'Q',1,'O')  # training data list top right 'Q' variant 1  
    
    trainingDataListQ6 =  (208,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],16,'Q',1,'O') # training data list bottom left 'Q' variant 2
    
    trainingDataListQ7 =  (209,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    ],16,'Q',1,'O')  # training data list bottom center 'Q' variant 2

    trainingDataListQ8 =  (210,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,  
    ],16,'Q',1,'O')  # training data list bottom right 'Q' variant 2  
    
    trainingDataListQ9 =  (211,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,1,0,0,1,0,0,
    1,0,0,0,0,0,1,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],16,'Q',1,'O')  # training data list top left 'Q' variant 2
    
    trainingDataListQ10 =  (212,[ 
    0,0,1,1,1,1,1,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,1,0,0,1,0,
    0,1,0,0,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,  
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],16,'Q',1,'O')  # training data list top center 'Q' variant 2

    trainingDataListQ11 =  (213,[
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,1,0,0,1,
    0,0,1,0,0,0,0,0,1,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,  
    ],16,'Q',1,'O') # training data list top right 'Q' variant 2             
                               
########
#  R   #
########                        

    trainingDataListR0 =  (18,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0, 
    ],17,'R',2,'F')  # training data list bottom left 'R' variant 1
    
    trainingDataListR1 =  (214,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0, 
    ],17,'R',2,'F')  # training data list bottom center 'R' variant 1

    trainingDataListR2 =  (215,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1, 
    ],17,'R',2,'F') # training data list bottom right 'R' variant 1 
    
    trainingDataListR3 =  (216,[ 
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,1,1,1,1,1,1,0,0,0,0, 
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],17,'R',2,'F')  # training data list top left 'R' variant 1
    
    trainingDataListR4 =  (217,[
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,1,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],17,'R',2,'F')  # training data list top center 'R' variant 1

    trainingDataListR5 =  (218,[ 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],17,'R',2,'F') # training data list top right 'R' variant 1 
                      
    trainingDataListR6 =  (219,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0, 
    ],17,'R',2,'F')  # training data list bottom left 'R' variant 2
    
    trainingDataListR7 =  (220,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0, 
    ],17,'R',2,'F')  # training data list bottom center 'R' variant 2

    trainingDataListR8 =  (221,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1, 
    ],17,'R',2,'F')  # training data list bottom right 'R' variant 2 
    
    trainingDataListR9 =  (222,[ 
    1,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,1,1,0,0,
    1,1,1,1,1,1,1,1,0,0,0,
    1,0,0,0,0,1,0,0,0,0,0,
    1,0,0,0,0,0,1,0,0,0,0,
    1,0,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],17,'R',2,'F')  # training data list top left 'R' variant 2
    
    trainingDataListR10 =  (223,[
    0,1,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,1,1,0,
    0,1,1,1,1,1,1,1,1,0,0,
    0,1,0,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,1,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],17,'R',2,'F')  # training data list top center 'R' variant 2

    trainingDataListR11 =  (224,[ 
    0,0,1,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,1,
    0,0,1,1,1,1,1,1,1,1,0,
    0,0,1,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],17,'R',2,'F') # training data list top right 'R' variant 2   
    
########
#  S  #
########                        

    trainingDataListS0 =  (19,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],18,'S',1,'O')  # training data list bottom left 'S' variant 1
    
    trainingDataListS1 =  (225,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    ],18,'S',1,'O')  # training data list bottom center 'S' variant 1

    trainingDataListS2 =  (226,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,1,1,1,1,1,1,0, 
    ],18,'S',1,'O') # training data list bottom right 'S' variant 1 
    
    trainingDataListS3 =  (227,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],18,'S',1,'O')  # training data list top left 'S' variant 1
    
    trainingDataListS4 =  (228,[ 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],18,'S',1,'O')  # training data list top center 'S' variant 1

    trainingDataListS5 =  (229,[ 
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],18,'S',1,'O') # training data list top right 'S' variant 1 
    
    trainingDataListS6 =  (230,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0, 
    ],18,'S',1,'O')  # training data list bottom left 'S' variant 2
    
    trainingDataListS7 =  (231,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,
    0,0,0,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    ],18,'S',1,'O')  # training data list bottom center 'S' variant 2

    trainingDataListS8 =  (232,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,1,1,1,0,
    0,0,0,0,0,0,0,0,0,1,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,
    ],18,'S',1,'O') # training data list bottom right 'S' variant 2 
    
    trainingDataListS9 =  (233,[
    0,1,1,1,1,1,1,1,0,0,0, 
    1,1,0,0,0,0,0,1,1,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],18,'S',1,'O')  # training data list top left 'S' variant 2
    
    trainingDataListS10 =  (234,[  
    0,0,1,1,1,1,1,1,1,0,0, 
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,0,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,
    0,0,0,0,0,0,0,0,1,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],18,'S',1,'O')  # training data list top center 'S' variant 2

    trainingDataListS11 =  (235,[ 
    0,0,0,1,1,1,1,1,1,1,0, 
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,1,1,1,0,
    0,0,0,0,0,0,0,0,0,1,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],18,'S',1,'O') # training data list top right 'S' variant 2 

########
#  T  #
########                        

    trainingDataListT0 =  (20,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0, 
    ],19,'T',2,'F')  # training data list bottom left 'T' variant 1
    
    trainingDataListT1 =  (236,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list bottom center 'T' variant 1

    trainingDataListT2 =  (237,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],19,'T',2,'F') # training data list bottom right 'T' variant 1   
    
    trainingDataListT3 =  (238,[ 
    1,1,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list top left 'T' variant 1
    
    trainingDataListT4 =  (239,[ 
    0,1,1,1,1,1,1,1,1,1,0, 
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list top center 'T' variant 1

    trainingDataListT5 =  (240,[ 
    0,0,1,1,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],19,'T',2,'F') # training data list top right 'T' variant 1
    
    trainingDataListT6 =  (241,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list bottom left 'T' variant 2
    
    trainingDataListT7 =  (242,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list bottom center 'T' variant 2

    trainingDataListT8 =  (243,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    ],19,'T',2,'F') # training data list bottom right 'T' variant 2   
    
    trainingDataListT9 =  (244,[ 
    1,1,1,1,1,1,1,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0, 
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],19,'T',2,'F')  # training data list top left 'T' variant 2
    
    trainingDataListT10 =  (245,[  
    0,0,1,1,1,1,1,1,1,0,0, 
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],19,'T',2,'F')  # training data list top center 'T' variant 2

    trainingDataListT11 =  (246,[ 
    0,0,0,0,1,1,1,1,1,1,1, 
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],19,'T',2,'F') # training data list top right 'T' variant 2


########
#  U  #
########                        

    trainingDataListU0 =  (21,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0,
    ],20,'U',4,'N')  # training data list bottom left 'U' variant 1
    
    trainingDataListU1 =  (247,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    ],20,'U',4,'N')  # training data list bottom center 'U' variant 1

    trainingDataListU2 =  (248,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,
    ],20,'U',4,'N') # training data list bottom right 'U' variant 1      
    
    trainingDataListU3 =  (249,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,1,1,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N')  # training data list top left 'U' variant 1
    
    trainingDataListU4 =  (250,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N')  # training data list top center 'U' variant 1

    trainingDataListU5 =  (251,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,1,0,
    0,0,0,0,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N') # training data list top right 'U' variant 1 
    
    trainingDataListU6 =  (252,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    ],20,'U',4,'N')  # training data list bottom left 'U' variant 2
    
    trainingDataListU7 =  (253,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,
    ],20,'U',4,'N')  # training data list bottom center 'U' variant 2

    trainingDataListU8 =  (254,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,
    ],20,'U',4,'N') # training data list bottom right 'U' variant 2      
    
    trainingDataListU9 =  (255,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,1,1,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N')  # training data list top left 'U' variant 2
    
    trainingDataListU10 =  (256,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N')  # training data list top center 'U' variant 2

    trainingDataListU11 =  (257,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],20,'U',4,'N') # training data list top right 'U' variant 2 
    
########
#  V  #
########                        

    trainingDataListV0 =  (22,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list bottom left 'V' variant 1
    
    trainingDataListV1 =  (258,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list bottom center 'V' variant 1

    trainingDataListV2 =  (259,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],21,'V',3,'X') # training data list bottom right 'V' variant 1   
    
    trainingDataListV3 =  (260,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list top left 'V' variant 1
    
    trainingDataListV4 =  (261,[
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list top center 'V' variant 1

    trainingDataListV5 =  (262,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X') # training data list top right 'V' variant 1     
    
    trainingDataListV6 =  (263,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list bottom left 'V' variant 2
    
    trainingDataListV7 =  (264,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list bottom center 'V' variant 2

    trainingDataListV8 =  (265,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],21,'V',3,'X') # training data list bottom right 'V' variant 2   
    
    trainingDataListV9 =  (266,[  
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,1,1,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],21,'V',3,'X')  # training data list top left 'V' variant 2
    
    trainingDataListV10 =  (267,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X')  # training data list top center 'V' variant 2

    trainingDataListV11 =  (268,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],21,'V',3,'X') # training data list top right 'V' variant 2   
    
    
########
#  W  #
########                        

    trainingDataListW0 =  (23,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,1,0,1,0,0,1,0,0,
    1,0,1,0,0,0,1,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],22,'W',4,'N')  # training data list bottom left 'W' variant 1
    
    trainingDataListW1 =  (269,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],22,'W',4,'N')  # training data list bottom center 'W' variant 1

    trainingDataListW2 =  (270,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,1,
    0,0,1,0,1,0,0,0,1,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],22,'W',4,'N') # training data list bottom right 'W' variant 1      
    
    trainingDataListW3 =  (271,[ 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,1,0,1,0,0,1,0,0,
    1,0,1,0,0,0,1,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N')  # training data list top left 'W' variant 1
    
    trainingDataListW4 =  (272,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N')  # training data list top center 'W' variant 1

    trainingDataListW5 =  (273,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,1,
    0,0,1,0,1,0,0,0,1,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N') # training data list top right 'W' variant 1  


    trainingDataListW6 =  (274,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,1,1,1,0,0,1,0,0,
    1,0,1,1,0,1,1,0,1,0,0,
    1,1,1,0,0,0,1,1,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    ],22,'W',4,'N')  # training data list bottom left 'W' variant 2
    
    trainingDataListW7 =  (275,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,1,1,1,0,0,1,0,
    0,1,0,1,1,0,1,1,0,1,0,
    0,1,1,1,0,0,0,1,1,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    ],22,'W',4,'N')  # training data list bottom center 'W' variant 2

    trainingDataListW8 =  (276,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,1,1,1,0,0,1,
    0,0,1,0,1,1,0,1,1,0,1,
    0,0,1,1,1,0,0,0,1,1,1,
    0,0,0,1,0,0,0,0,0,1,0,
    ],22,'W',4,'N') # training data list bottom right 'W' variant 2      
    
    trainingDataListW9 =  (277,[  
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    1,0,0,0,1,0,0,0,1,0,0,
    1,0,0,1,1,1,0,0,1,0,0,
    1,0,1,1,0,1,1,0,1,0,0,
    1,1,1,0,0,0,1,1,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N')  # training data list top left 'W' variant 2
    
    trainingDataListW10 =  (278,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,0,0,0,1,0,0,0,1,0,
    0,1,0,0,1,0,1,0,0,1,0,
    0,1,0,1,0,0,0,1,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],22,'W',4,'N')  # training data list top center 'W' variant 2

    trainingDataListW11 =  (279,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,0,0,0,1,0,0,0,1,
    0,0,1,0,0,1,1,1,0,0,1,
    0,0,1,0,1,1,0,1,1,0,1,
    0,0,1,1,1,0,0,0,1,1,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],22,'W',4,'N') # training data list top right 'W' variant 2  

########
#  X  #
########                        

    trainingDataListX0 =  (24,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],23,'X',3,'X')  # training data list bottom left 'X' variant 1
    
    trainingDataListX1 =  (280,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],23,'X',3,'X')  # training data list bottom center 'X' variant 1

    trainingDataListX2 =  (281,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    ],23,'X',3,'X') # training data list bottom right 'X' variant 1  
    
    trainingDataListX3 =  (282,[
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],23,'X',3,'X')  # training data list top left 'X' variant 1
    
    trainingDataListX4 =  (283,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],23,'X',3,'X')  # training data list top center 'X' variant 1

    trainingDataListX5 =  (284,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],23,'X',3,'X') # training data list top right 'X' variant 1  
    
    trainingDataListX6 =  (285,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    ],23,'X',3,'X')  # training data list bottom left 'X' variant 2
    
    trainingDataListX7 =  (286,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    ],23,'X',3,'X')  # training data list bottom center 'X' variant 2

    trainingDataListX8 =  (287,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1,
    ],23,'X',3,'X') # training data list bottom right 'X' variant 2 
    
    trainingDataListX9 =  (288,[
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    1,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],23,'X',3,'X')  # training data list top left 'X' variant 2
    
    trainingDataListX10 =  (289,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],23,'X',3,'X')  # training data list top center 'X' variant 2

    trainingDataListX11 =  (290,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,1,0,0,0,0,0,0,0,1, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],23,'X',3,'X') # training data list top right 'X' variant 2  
    
########
#  Y  #
########                        

    trainingDataListY0 =  (25,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    ],24,'Y',3,'X')   # training data list bottom left 'Y' variant 1
    
    trainingDataListY1 =  (291,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],24,'Y',3,'X')  # training data list bottom center 'Y' variant 1

    trainingDataListY2 =  (292,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],24,'Y',3,'X') # training data list bottom right 'Y' variant 1  
    
    trainingDataListY3 =  (293,[
    1,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,1,0,0,0,
    0,0,1,0,0,0,1,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],24,'Y',3,'X')  # training data list top left 'Y' variant 1
    
    trainingDataListY4 =  (294,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,0,1,0,0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,1,0,0,0,
    0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],24,'Y',3,'X')  # training data list top center 'Y variant 1

    trainingDataListY5 =  (295,[
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,0,1,0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,0,1,0,0,
    0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],24,'Y',3,'X') # training data list top right 'Y' variant 1  
    
    trainingDataListY6 =  (296,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    ],24,'Y',3,'X')  # training data list bottom left 'Y' variant 2
    
    trainingDataListY7 =  (297,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    ],24,'Y',3,'X')  # training data list bottom center 'Y' variant 2

    trainingDataListY8 =  (298,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    ],24,'Y',3,'X') # training data list bottom right 'Y' variant 2 
    
    trainingDataListY9 =  (299,[
    1,0,0,0,0,0,0,0,1,0,0,
    1,1,0,0,0,0,0,1,1,0,0,
    0,1,1,0,0,0,1,1,0,0,0,
    0,0,1,1,0,1,1,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],24,'Y',3,'X')  # training data list top left 'Y' variant 2
    
    trainingDataListY10 =  (300,[ 
    0,1,0,0,0,0,0,0,0,1,0,
    0,1,1,0,0,0,0,0,1,1,0,
    0,0,1,1,0,0,0,1,1,0,0,
    0,0,0,1,1,0,1,1,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],24,'Y',3,'X')  # training data list top center 'Y' variant 2

    trainingDataListY11 =  (301,[ 
    0,0,1,0,0,0,0,0,0,0,1,
    0,0,1,1,0,0,0,0,0,1,1,
    0,0,0,1,1,0,0,0,1,1,0,
    0,0,0,0,1,1,0,1,1,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],24,'Y',3,'X') # training data list top right 'Y' variant 2
  
########
#  Z  #
########                        

    trainingDataListZ0 =  (26,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    ],25,'Z',0,'L')    # training data list bottom left 'Z' variant 1
    
    trainingDataListZ1 =  (302,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    ],25,'Z',0,'L')   # training data list bottom center 'Z' variant 1

    trainingDataListZ2 =  (303,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    ],25,'Z',0,'L')  # training data list bottom right 'Z' variant 1  
    
    trainingDataListZ3 =  (304,[ 
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],25,'Z',0,'L')    # training data list top left 'Z' variant 1
    
    trainingDataListZ4 =  (305,[
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0
    ],25,'Z',0,'L')   # training data list top center 'Z' variant 1

    trainingDataListZ5 =  (306,[
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],25,'Z',0,'L')  # training data list top right 'Z' variant 1  
    
    trainingDataListZ6 =  (307,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    ],25,'Z',0,'L')    # training data list bottom left 'Z' variant 2
    
    trainingDataListZ7 =  (308,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    ],25,'Z',0,'L')   # training data list bottom center 'Z' variant 2

    trainingDataListZ8 =  (309,[
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    ],25,'Z',0,'L')  # training data list bottom right 'Z' variant 2  
    
    trainingDataListZ9 =  (310,[ 
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0,
    ],25,'Z',0,'L')    # training data list top left 'Z' variant 2
    
    trainingDataListZ10 =  (311,[
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,1,1,0,0,0,0,0,0,0,0,
    0,1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0
    ],25,'Z',0,'L')   # training data list top center 'Z' variant 2

    trainingDataListZ11 =  (312,[
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,1,1,0,
    0,0,0,0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,0,
    0,0,0,0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0, 
    0,0,0,0,0,0,0,0,0,0,0, 
    ],25,'Z',0,'L')  # training data list top right 'Z' variant 2 

       
    if dataSet == 1: trainingDataList = trainingDataListA0
    if dataSet == 2: trainingDataList = trainingDataListA1    
    if dataSet == 3: trainingDataList = trainingDataListA2
    if dataSet == 4: trainingDataList = trainingDataListA3
    if dataSet == 5: trainingDataList = trainingDataListA4        
    if dataSet == 6: trainingDataList = trainingDataListA5
    if dataSet == 7: trainingDataList = trainingDataListA6
    if dataSet == 8: trainingDataList = trainingDataListA7   
    if dataSet == 9: trainingDataList = trainingDataListA8 
    if dataSet == 10: trainingDataList = trainingDataListA9
    if dataSet == 11: trainingDataList = trainingDataListA10  
    if dataSet == 12: trainingDataList = trainingDataListA11 
                                                                                                 
    if dataSet == 13: trainingDataList = trainingDataListB0
    if dataSet == 14: trainingDataList = trainingDataListB1    
    if dataSet == 15: trainingDataList = trainingDataListB2
    if dataSet == 16: trainingDataList = trainingDataListB3
    if dataSet == 17: trainingDataList = trainingDataListB4        
    if dataSet == 18: trainingDataList = trainingDataListB5
    if dataSet == 19: trainingDataList = trainingDataListB6
    if dataSet == 20: trainingDataList = trainingDataListB7   
    if dataSet == 21: trainingDataList = trainingDataListB8 
    if dataSet == 22: trainingDataList = trainingDataListB9
    if dataSet == 23: trainingDataList = trainingDataListB10  
    if dataSet == 24: trainingDataList = trainingDataListB11 
    
    if dataSet == 25: trainingDataList = trainingDataListC0
    if dataSet == 26: trainingDataList = trainingDataListC1    
    if dataSet == 27: trainingDataList = trainingDataListC2
    if dataSet == 28: trainingDataList = trainingDataListC3
    if dataSet == 29: trainingDataList = trainingDataListC4        
    if dataSet == 30: trainingDataList = trainingDataListC5
    if dataSet == 31: trainingDataList = trainingDataListC6
    if dataSet == 32: trainingDataList = trainingDataListC7   
    if dataSet == 33: trainingDataList = trainingDataListC8 
    if dataSet == 34: trainingDataList = trainingDataListC9
    if dataSet == 35: trainingDataList = trainingDataListC10  
    if dataSet == 36: trainingDataList = trainingDataListC11 
    
    if dataSet == 37: trainingDataList = trainingDataListD0
    if dataSet == 38: trainingDataList = trainingDataListD1    
    if dataSet == 39: trainingDataList = trainingDataListD2
    if dataSet == 40: trainingDataList = trainingDataListD3
    if dataSet == 41: trainingDataList = trainingDataListD4        
    if dataSet == 42: trainingDataList = trainingDataListD5
    if dataSet == 43: trainingDataList = trainingDataListD6
    if dataSet == 44: trainingDataList = trainingDataListD7   
    if dataSet == 45: trainingDataList = trainingDataListD8 
    if dataSet == 46: trainingDataList = trainingDataListD9
    if dataSet == 47: trainingDataList = trainingDataListD10  
    if dataSet == 48: trainingDataList = trainingDataListD11 
    
    if dataSet == 49: trainingDataList = trainingDataListE0
    if dataSet == 50: trainingDataList = trainingDataListE1    
    if dataSet == 51: trainingDataList = trainingDataListE2
    if dataSet == 52: trainingDataList = trainingDataListE3
    if dataSet == 53: trainingDataList = trainingDataListE4        
    if dataSet == 54: trainingDataList = trainingDataListE5
    if dataSet == 55: trainingDataList = trainingDataListE6
    if dataSet == 56: trainingDataList = trainingDataListE7   
    if dataSet == 57: trainingDataList = trainingDataListE8 
    if dataSet == 58: trainingDataList = trainingDataListE9
    if dataSet == 59: trainingDataList = trainingDataListE10  
    if dataSet == 60: trainingDataList = trainingDataListE11 
    
    if dataSet == 61: trainingDataList = trainingDataListF0
    if dataSet == 62: trainingDataList = trainingDataListF1    
    if dataSet == 63: trainingDataList = trainingDataListF2
    if dataSet == 64: trainingDataList = trainingDataListF3
    if dataSet == 65: trainingDataList = trainingDataListF4        
    if dataSet == 66: trainingDataList = trainingDataListF5
    if dataSet == 67: trainingDataList = trainingDataListF6
    if dataSet == 68: trainingDataList = trainingDataListF7   
    if dataSet == 69: trainingDataList = trainingDataListF8 
    if dataSet == 70: trainingDataList = trainingDataListF9
    if dataSet == 71: trainingDataList = trainingDataListF10  
    if dataSet == 72: trainingDataList = trainingDataListF11 
    
    if dataSet == 73: trainingDataList = trainingDataListG0
    if dataSet == 74: trainingDataList = trainingDataListG1    
    if dataSet == 75: trainingDataList = trainingDataListG2
    if dataSet == 76: trainingDataList = trainingDataListG3
    if dataSet == 77: trainingDataList = trainingDataListG4        
    if dataSet == 78: trainingDataList = trainingDataListG5
    if dataSet == 79: trainingDataList = trainingDataListG6
    if dataSet == 80: trainingDataList = trainingDataListG7   
    if dataSet == 81: trainingDataList = trainingDataListG8 
    if dataSet == 82: trainingDataList = trainingDataListG9
    if dataSet == 83: trainingDataList = trainingDataListG10  
    if dataSet == 84: trainingDataList = trainingDataListG11     
    
    if dataSet == 85: trainingDataList = trainingDataListH0
    if dataSet == 86: trainingDataList = trainingDataListH1    
    if dataSet == 87: trainingDataList = trainingDataListH2
    if dataSet == 88: trainingDataList = trainingDataListH3
    if dataSet == 89: trainingDataList = trainingDataListH4        
    if dataSet == 90: trainingDataList = trainingDataListH5
    if dataSet == 91: trainingDataList = trainingDataListH6
    if dataSet == 92: trainingDataList = trainingDataListH7   
    if dataSet == 93: trainingDataList = trainingDataListH8 
    if dataSet == 94: trainingDataList = trainingDataListH9
    if dataSet == 95: trainingDataList = trainingDataListH10  
    if dataSet == 96: trainingDataList = trainingDataListH11                                                                                                                                                                                                                
    
    if dataSet == 97: trainingDataList = trainingDataListI0
    if dataSet == 98: trainingDataList = trainingDataListI1    
    if dataSet == 99: trainingDataList = trainingDataListI2
    if dataSet == 100: trainingDataList = trainingDataListI3
    if dataSet == 101: trainingDataList = trainingDataListI4        
    if dataSet == 102: trainingDataList = trainingDataListI5
    if dataSet == 103: trainingDataList = trainingDataListI6
    if dataSet == 104: trainingDataList = trainingDataListI7   
    if dataSet == 105: trainingDataList = trainingDataListI8 
    if dataSet == 106: trainingDataList = trainingDataListI9
    if dataSet == 107: trainingDataList = trainingDataListI10  
    if dataSet == 108: trainingDataList = trainingDataListI11       
    
    if dataSet == 109: trainingDataList = trainingDataListJ0
    if dataSet == 110: trainingDataList = trainingDataListJ1    
    if dataSet == 111: trainingDataList = trainingDataListJ2
    if dataSet == 112: trainingDataList = trainingDataListJ3
    if dataSet == 113: trainingDataList = trainingDataListJ4        
    if dataSet == 114: trainingDataList = trainingDataListJ5
    if dataSet == 115: trainingDataList = trainingDataListJ6
    if dataSet == 116: trainingDataList = trainingDataListJ7   
    if dataSet == 117: trainingDataList = trainingDataListJ8 
    if dataSet == 118: trainingDataList = trainingDataListJ9
    if dataSet == 119: trainingDataList = trainingDataListJ10  
    if dataSet == 120: trainingDataList = trainingDataListJ11   
    
    if dataSet == 121: trainingDataList = trainingDataListK0
    if dataSet == 122: trainingDataList = trainingDataListK1    
    if dataSet == 123: trainingDataList = trainingDataListK2
    if dataSet == 124: trainingDataList = trainingDataListK3
    if dataSet == 125: trainingDataList = trainingDataListK4        
    if dataSet == 126: trainingDataList = trainingDataListK5
    if dataSet == 127: trainingDataList = trainingDataListK6
    if dataSet == 128: trainingDataList = trainingDataListK7   
    if dataSet == 129: trainingDataList = trainingDataListK8 
    if dataSet == 130: trainingDataList = trainingDataListK9
    if dataSet == 131: trainingDataList = trainingDataListK10  
    if dataSet == 132: trainingDataList = trainingDataListK11     
    
    if dataSet == 133: trainingDataList = trainingDataListL0
    if dataSet == 134: trainingDataList = trainingDataListL1    
    if dataSet == 135: trainingDataList = trainingDataListL2
    if dataSet == 136: trainingDataList = trainingDataListL3
    if dataSet == 137: trainingDataList = trainingDataListL4        
    if dataSet == 138: trainingDataList = trainingDataListL5
    if dataSet == 139: trainingDataList = trainingDataListL6
    if dataSet == 140: trainingDataList = trainingDataListL7   
    if dataSet == 141: trainingDataList = trainingDataListL8 
    if dataSet == 142: trainingDataList = trainingDataListL9
    if dataSet == 143: trainingDataList = trainingDataListL10  
    if dataSet == 144: trainingDataList = trainingDataListL11     
    
    if dataSet == 145: trainingDataList = trainingDataListM0
    if dataSet == 146: trainingDataList = trainingDataListM1    
    if dataSet == 147: trainingDataList = trainingDataListM2
    if dataSet == 148: trainingDataList = trainingDataListM3
    if dataSet == 149: trainingDataList = trainingDataListM4        
    if dataSet == 150: trainingDataList = trainingDataListM5
    if dataSet == 151: trainingDataList = trainingDataListM6
    if dataSet == 152: trainingDataList = trainingDataListM7   
    if dataSet == 153: trainingDataList = trainingDataListM8 
    if dataSet == 154: trainingDataList = trainingDataListM9
    if dataSet == 155: trainingDataList = trainingDataListM10  
    if dataSet == 156: trainingDataList = trainingDataListM11    
    
    if dataSet == 157: trainingDataList = trainingDataListN0
    if dataSet == 158: trainingDataList = trainingDataListN1    
    if dataSet == 159: trainingDataList = trainingDataListN2
    if dataSet == 160: trainingDataList = trainingDataListN3
    if dataSet == 161: trainingDataList = trainingDataListN4        
    if dataSet == 162: trainingDataList = trainingDataListN5
    if dataSet == 163: trainingDataList = trainingDataListN6
    if dataSet == 164: trainingDataList = trainingDataListN7   
    if dataSet == 165: trainingDataList = trainingDataListN8 
    if dataSet == 166: trainingDataList = trainingDataListN9
    if dataSet == 167: trainingDataList = trainingDataListN10  
    if dataSet == 168: trainingDataList = trainingDataListN11     
    
    if dataSet == 169: trainingDataList = trainingDataListO0
    if dataSet == 170: trainingDataList = trainingDataListO1    
    if dataSet == 171: trainingDataList = trainingDataListO2
    if dataSet == 172: trainingDataList = trainingDataListO3
    if dataSet == 173: trainingDataList = trainingDataListO4        
    if dataSet == 174: trainingDataList = trainingDataListO5
    if dataSet == 175: trainingDataList = trainingDataListO6
    if dataSet == 176: trainingDataList = trainingDataListO7   
    if dataSet == 177: trainingDataList = trainingDataListO8 
    if dataSet == 178: trainingDataList = trainingDataListO9
    if dataSet == 179: trainingDataList = trainingDataListO10  
    if dataSet == 180: trainingDataList = trainingDataListO11    
    
    if dataSet == 181: trainingDataList = trainingDataListP0
    if dataSet == 182: trainingDataList = trainingDataListP1    
    if dataSet == 183: trainingDataList = trainingDataListP2
    if dataSet == 184: trainingDataList = trainingDataListP3
    if dataSet == 185: trainingDataList = trainingDataListP4        
    if dataSet == 186: trainingDataList = trainingDataListP5
    if dataSet == 187: trainingDataList = trainingDataListP6
    if dataSet == 188: trainingDataList = trainingDataListP7   
    if dataSet == 189: trainingDataList = trainingDataListP8 
    if dataSet == 190: trainingDataList = trainingDataListP9
    if dataSet == 191: trainingDataList = trainingDataListP10  
    if dataSet == 192: trainingDataList = trainingDataListP11    
    
    if dataSet == 193: trainingDataList = trainingDataListQ0
    if dataSet == 194: trainingDataList = trainingDataListQ1    
    if dataSet == 195: trainingDataList = trainingDataListQ2
    if dataSet == 196: trainingDataList = trainingDataListQ3
    if dataSet == 197: trainingDataList = trainingDataListQ4        
    if dataSet == 198: trainingDataList = trainingDataListQ5
    if dataSet == 199: trainingDataList = trainingDataListQ6
    if dataSet == 200: trainingDataList = trainingDataListQ7   
    if dataSet == 201: trainingDataList = trainingDataListQ8 
    if dataSet == 202: trainingDataList = trainingDataListQ9
    if dataSet == 203: trainingDataList = trainingDataListQ10  
    if dataSet == 204: trainingDataList = trainingDataListQ11  
    
    if dataSet == 205: trainingDataList = trainingDataListR0
    if dataSet == 206: trainingDataList = trainingDataListR1    
    if dataSet == 207: trainingDataList = trainingDataListR2
    if dataSet == 208: trainingDataList = trainingDataListR3
    if dataSet == 209: trainingDataList = trainingDataListR4        
    if dataSet == 210: trainingDataList = trainingDataListR5
    if dataSet == 211: trainingDataList = trainingDataListR6
    if dataSet == 212: trainingDataList = trainingDataListR7   
    if dataSet == 213: trainingDataList = trainingDataListR8 
    if dataSet == 214: trainingDataList = trainingDataListR9
    if dataSet == 215: trainingDataList = trainingDataListR10  
    if dataSet == 216: trainingDataList = trainingDataListR11 
    
    if dataSet == 217: trainingDataList = trainingDataListS0
    if dataSet == 218: trainingDataList = trainingDataListS1    
    if dataSet == 219: trainingDataList = trainingDataListS2
    if dataSet == 220: trainingDataList = trainingDataListS3
    if dataSet == 221: trainingDataList = trainingDataListS4        
    if dataSet == 222: trainingDataList = trainingDataListS5
    if dataSet == 223: trainingDataList = trainingDataListS6
    if dataSet == 224: trainingDataList = trainingDataListS7   
    if dataSet == 225: trainingDataList = trainingDataListS8 
    if dataSet == 226: trainingDataList = trainingDataListS9
    if dataSet == 227: trainingDataList = trainingDataListS10  
    if dataSet == 228: trainingDataList = trainingDataListS11     
    
    if dataSet == 229: trainingDataList = trainingDataListT0
    if dataSet == 230: trainingDataList = trainingDataListT1    
    if dataSet == 231: trainingDataList = trainingDataListT2
    if dataSet == 232: trainingDataList = trainingDataListT3
    if dataSet == 233: trainingDataList = trainingDataListT4        
    if dataSet == 234: trainingDataList = trainingDataListT5
    if dataSet == 235: trainingDataList = trainingDataListT6
    if dataSet == 236: trainingDataList = trainingDataListT7   
    if dataSet == 237: trainingDataList = trainingDataListT8 
    if dataSet == 238: trainingDataList = trainingDataListT9
    if dataSet == 239: trainingDataList = trainingDataListT10  
    if dataSet == 240: trainingDataList = trainingDataListT11        
    
    if dataSet == 241: trainingDataList = trainingDataListU0
    if dataSet == 242: trainingDataList = trainingDataListU1    
    if dataSet == 243: trainingDataList = trainingDataListU2
    if dataSet == 244: trainingDataList = trainingDataListU3
    if dataSet == 245: trainingDataList = trainingDataListU4        
    if dataSet == 246: trainingDataList = trainingDataListU5
    if dataSet == 247: trainingDataList = trainingDataListU6
    if dataSet == 248: trainingDataList = trainingDataListU7   
    if dataSet == 249: trainingDataList = trainingDataListU8 
    if dataSet == 250: trainingDataList = trainingDataListU9
    if dataSet == 251: trainingDataList = trainingDataListU10  
    if dataSet == 252: trainingDataList = trainingDataListU11            
    
    if dataSet == 253: trainingDataList = trainingDataListV0
    if dataSet == 254: trainingDataList = trainingDataListV1    
    if dataSet == 255: trainingDataList = trainingDataListV2
    if dataSet == 256: trainingDataList = trainingDataListV3
    if dataSet == 257: trainingDataList = trainingDataListV4        
    if dataSet == 258: trainingDataList = trainingDataListV5
    if dataSet == 259: trainingDataList = trainingDataListV6
    if dataSet == 260: trainingDataList = trainingDataListV7   
    if dataSet == 261: trainingDataList = trainingDataListV8 
    if dataSet == 262: trainingDataList = trainingDataListV9
    if dataSet == 263: trainingDataList = trainingDataListV10  
    if dataSet == 264: trainingDataList = trainingDataListV11                 
    
    if dataSet == 265: trainingDataList = trainingDataListW0
    if dataSet == 266: trainingDataList = trainingDataListW1    
    if dataSet == 267: trainingDataList = trainingDataListW2
    if dataSet == 268: trainingDataList = trainingDataListW3
    if dataSet == 269: trainingDataList = trainingDataListW4        
    if dataSet == 270: trainingDataList = trainingDataListW5
    if dataSet == 271: trainingDataList = trainingDataListW6
    if dataSet == 272: trainingDataList = trainingDataListW7   
    if dataSet == 273: trainingDataList = trainingDataListW8 
    if dataSet == 274: trainingDataList = trainingDataListW9
    if dataSet == 275: trainingDataList = trainingDataListW10  
    if dataSet == 276: trainingDataList = trainingDataListW11 
    
    if dataSet == 277: trainingDataList = trainingDataListX0
    if dataSet == 278: trainingDataList = trainingDataListX1    
    if dataSet == 279: trainingDataList = trainingDataListX2
    if dataSet == 280: trainingDataList = trainingDataListX3
    if dataSet == 281: trainingDataList = trainingDataListX4        
    if dataSet == 282: trainingDataList = trainingDataListX5
    if dataSet == 283: trainingDataList = trainingDataListX6
    if dataSet == 284: trainingDataList = trainingDataListX7   
    if dataSet == 285: trainingDataList = trainingDataListX8 
    if dataSet == 286: trainingDataList = trainingDataListX9
    if dataSet == 287: trainingDataList = trainingDataListX10  
    if dataSet == 288: trainingDataList = trainingDataListX11 
    
    if dataSet == 289: trainingDataList = trainingDataListY0
    if dataSet == 290: trainingDataList = trainingDataListY1    
    if dataSet == 291: trainingDataList = trainingDataListY2
    if dataSet == 292: trainingDataList = trainingDataListY3
    if dataSet == 293: trainingDataList = trainingDataListY4        
    if dataSet == 294: trainingDataList = trainingDataListY5
    if dataSet == 295: trainingDataList = trainingDataListY6
    if dataSet == 296: trainingDataList = trainingDataListY7   
    if dataSet == 297: trainingDataList = trainingDataListY8 
    if dataSet == 298: trainingDataList = trainingDataListY9
    if dataSet == 299: trainingDataList = trainingDataListY10  
    if dataSet == 300: trainingDataList = trainingDataListY11 
    
    if dataSet == 301: trainingDataList = trainingDataListZ0
    if dataSet == 302: trainingDataList = trainingDataListZ1    
    if dataSet == 303: trainingDataList = trainingDataListZ2
    if dataSet == 304: trainingDataList = trainingDataListZ3
    if dataSet == 305: trainingDataList = trainingDataListZ4        
    if dataSet == 306: trainingDataList = trainingDataListZ5
    if dataSet == 307: trainingDataList = trainingDataListZ6
    if dataSet == 308: trainingDataList = trainingDataListZ7   
    if dataSet == 309: trainingDataList = trainingDataListZ8 
    if dataSet == 310: trainingDataList = trainingDataListZ9
    if dataSet == 311: trainingDataList = trainingDataListZ10  
    if dataSet == 312: trainingDataList = trainingDataListZ11   
                                         
                                                                                        
    return (trainingDataList)  
           
####################################################################################################
####################################################################################################
#
# Perform a single feedforward pass
#
####################################################################################################
####################################################################################################



####################################################################################################
#
# Grey Box 1 Function to compute the GB1 hidden node activations as first part of a feedforward pass and return
#   the results in GB1hiddenArray
#
####################################################################################################


def ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray, GB1wWeightArray, GB1wBiasWeightArray):     
            
# iniitalize the sum of inputs into the hidden array with 0's  
    GB1sumIntoHiddenArray = np.zeros(GB1hiddenArrayLength)    
    GB1hiddenArray = np.zeros(GB1hiddenArrayLength)   

    GB1sumIntoHiddenArray = matrixDotProduct (GB1wWeightArray,GB1inputDataArray)
    
    for node in range(GB1hiddenArrayLength):  #  Number of hidden nodes
        GB1hiddenNodeSumInput=GB1sumIntoHiddenArray[node]+GB1wBiasWeightArray[node]
        GB1hiddenArray[node] = computeTransferFnctn(GB1hiddenNodeSumInput, alpha)
                                                                                                    
    return (GB1hiddenArray);
  


####################################################################################################
#
# Grey Box 1 Function to compute the output node activations, given the GB1 hidden node activations, 
#   the GB1 hidden-to output connection weights, and the GB1 output bias weights.
# Function returns the array of GB1 output node activations for Grey Box 1.
#
####################################################################################################

def ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray, GB1vWeightArray, GB1vBiasWeightArray):
    
# initialize the sum of inputs into the hidden array with 0's  
    GB1sumIntoOutputArray = np.zeros(GB1hiddenArrayLength)    
    GB1outputArray = np.zeros(GB1outputArrayLength)   

    GB1sumIntoOutputArray = matrixDotProduct (GB1vWeightArray,GB1hiddenArray)
    
    for node in range(GB1outputArrayLength):  #  Number of hidden nodes
        GB1outputNodeSumInput=GB1sumIntoOutputArray[node]+GB1vBiasWeightArray[node]
        GB1outputArray[node] = computeTransferFnctn(GB1outputNodeSumInput, alpha)
                                                                                                   
    return (GB1outputArray);
  



####################################################################################################
#
# Function to compute the hidden node activations as first part of a feedforward pass and return
#   the results in hiddenArray
#
####################################################################################################


def ComputeSingleFeedforwardPassFirstStep0 (alpha, inputDataList, wWeightArray0, biasHiddenWeightArray0):
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoHiddenArray0 = np.zeros(hiddenArrayLength)    
    hiddenArray0 = np.zeros(hiddenArrayLength)   



    sumIntoHiddenArray0 = matrixDotProduct (wWeightArray0,inputDataList)
    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        hiddenNodeSumInput0=sumIntoHiddenArray0[node]+biasHiddenWeightArray0[node]
        hiddenArray0[node] = computeTransferFnctn(hiddenNodeSumInput0, alpha)
                                                                                                    
    return (hiddenArray0);
  
def ComputeSingleFeedforwardPassFirstStep1 (alpha, inputDataList, wWeightArray1, biasHiddenWeightArray1):
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoHiddenArray1 = np.zeros(hiddenArrayLength)    
    hiddenArray1 = np.zeros(hiddenArrayLength)   


    sumIntoHiddenArray1 = matrixDotProduct (wWeightArray1,inputDataList)
    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        hiddenNodeSumInput1=sumIntoHiddenArray1[node]+biasHiddenWeightArray1[node]
        hiddenArray1[node] = computeTransferFnctn(hiddenNodeSumInput1, alpha)
                                                                                                    
    return (hiddenArray1);
    
def ComputeSingleFeedforwardPassFirstStep2 (alpha, inputDataList, wWeightArray2, biasHiddenWeightArray2):
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoHiddenArray2 = np.zeros(hiddenArrayLength)    
    hiddenArray2 = np.zeros(hiddenArrayLength)   

    sumIntoHiddenArray2 = matrixDotProduct (wWeightArray2,inputDataList)
    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        hiddenNodeSumInput2=sumIntoHiddenArray2[node]+biasHiddenWeightArray2[node]
        hiddenArray2[node] = computeTransferFnctn(hiddenNodeSumInput2, alpha)
                                                                                                    
    return (hiddenArray2);
    
def ComputeSingleFeedforwardPassFirstStep3 (alpha, inputDataList, wWeightArray3, biasHiddenWeightArray3):
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoHiddenArray3 = np.zeros(hiddenArrayLength)    
    hiddenArray3 = np.zeros(hiddenArrayLength)   

    sumIntoHiddenArray3 = matrixDotProduct (wWeightArray3,inputDataList)
    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        hiddenNodeSumInput3=sumIntoHiddenArray3[node]+biasHiddenWeightArray3[node]
        hiddenArray3[node] = computeTransferFnctn(hiddenNodeSumInput3, alpha)
                                                                                                    
    return (hiddenArray3);
    
def ComputeSingleFeedforwardPassFirstStep4 (alpha, inputDataList, wWeightArray4, biasHiddenWeightArray4):
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoHiddenArray4 = np.zeros(hiddenArrayLength)    
    hiddenArray4 = np.zeros(hiddenArrayLength)   

    sumIntoHiddenArray4 = matrixDotProduct (wWeightArray4,inputDataList)
    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        hiddenNodeSumInput4=sumIntoHiddenArray4[node]+biasHiddenWeightArray4[node]
        hiddenArray4[node] = computeTransferFnctn(hiddenNodeSumInput4, alpha)
                                                                                                    
    return (hiddenArray4);

####################################################################################################
#
# Function to compute the output node activations, given the hidden node activations, the hidden-to
#   output connection weights, and the output bias weights.
# Function returns the array of output node activations.
#
####################################################################################################

def ComputeSingleFeedforwardPassSecondStep0 (alpha, hiddenArray0, vWeightArray0, biasOutputWeightArray0):
    
# initialize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray0 = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray0 = matrixDotProduct (vWeightArray0,hiddenArray0)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput0=sumIntoOutputArray0[node]+biasOutputWeightArray0[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput0, alpha)
                                                                                                   
    return (outputArray);
  
def ComputeSingleFeedforwardPassSecondStep1 (alpha, hiddenArray1, vWeightArray1, biasOutputWeightArray1):
    
# initialize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray1 = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray1 = matrixDotProduct (vWeightArray1,hiddenArray1)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput1=sumIntoOutputArray1[node]+biasOutputWeightArray1[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput1, alpha)
                                                                                                   
    return (outputArray);
    
def ComputeSingleFeedforwardPassSecondStep2 (alpha, hiddenArray2, vWeightArray2, biasOutputWeightArray2):
    
# initialize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray2 = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray2 = matrixDotProduct (vWeightArray2,hiddenArray2)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput2=sumIntoOutputArray2[node]+biasOutputWeightArray2[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput2, alpha)
                                                                                                   
    return (outputArray);
    
def ComputeSingleFeedforwardPassSecondStep3 (alpha, hiddenArray3, vWeightArray3, biasOutputWeightArray3):
    
# initialize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray3 = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray3 = matrixDotProduct (vWeightArray3,hiddenArray3)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput3=sumIntoOutputArray3[node]+biasOutputWeightArray3[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput3, alpha)
                                                                                                   
    return (outputArray);
    
def ComputeSingleFeedforwardPassSecondStep4 (alpha, hiddenArray4, vWeightArray4, biasOutputWeightArray4):
    
# initialize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray4 = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray4 = matrixDotProduct (vWeightArray4,hiddenArray4)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput4=sumIntoOutputArray4[node]+biasOutputWeightArray4[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput4, alpha)
                                                                                                   
    return (outputArray);

####################################################################################################
#
# Procedure to compute the output node activations and determine errors across the entire training
#  data set, and print results.
#
####################################################################################################

def ComputeOutputsAcrossAllTrainingData (alpha, numTrainingDataSets, wWeightArray0, 
biasHiddenWeightArray0, vWeightArray0, biasOutputWeightArray0, wWeightArray1, 
biasHiddenWeightArray1, vWeightArray1, biasOutputWeightArray1, wWeightArray2, 
biasHiddenWeightArray2, vWeightArray2, biasOutputWeightArray2,wWeightArray3, 
biasHiddenWeightArray3, vWeightArray3, biasOutputWeightArray3,wWeightArray4, 
biasHiddenWeightArray4, vWeightArray4, biasOutputWeightArray4, GB1wWeightArray, GB1wBiasWeightArray, 
GB1vWeightArray, GB1vBiasWeightArray):

    selectedTrainingDataSet = 1                              
    newTotalSSE = 0.0 
    misclassification = 0.0                              

    while selectedTrainingDataSet < numTrainingDataSets + 1: 

        trainingDataList = obtainSelectedAlphabetTrainingValues (selectedTrainingDataSet)

        trainingDataInputList = trainingDataList[1]      

# Obtain the outputs from GB1
            
        GB1inputDataList = [] 
        GB1inputDataArray = np.zeros(GB1inputArrayLength)
         
        for node in range(GB1inputArrayLength): 
            trainingData = trainingDataInputList[node]  
            GB1inputDataList.append(trainingData)
            GB1inputDataArray[node] = trainingData

        GB1hiddenArray = ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray, GB1wWeightArray, GB1wBiasWeightArray)
        GB1outputArray = ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray, GB1vWeightArray, GB1vBiasWeightArray)                        
                                                                                                                                                                                                                                                                     
                                                                                
# The next step will be to create a padded version of this letter
#    (Expand boundaries by two pixels all around)
        expandedLetterArray = list()

        ExpandTrainingDataList = trainingDataList[1]
        expandedLetterArray = expandLetterBoundaries (ExpandTrainingDataList)

# Optional print/debug
    
        mask1LetterArray = maskLetterFunc(expandedLetterArray, mask1)
        mask1LetterList = convertArrayToList(mask1LetterArray)

        mask2LetterArray = maskLetterFunc(expandedLetterArray, mask2)
        mask2LetterList = convertArrayToList(mask2LetterArray)

                    

# Obtain the outputs from the full multi-component network

# First, obtain a full input vect
        inputDataList = [] 
        inputDataArray = np.zeros(inputArrayLength) 
      
#  First part of input is the original input list

        inputDataList = GB1inputDataList


#  Now, we're going to do that three more times ... same Masking Field, we're just filling in all 
#   the elements of the input data list. (We'll create new Masking Fields in the next round of code.)
        for node in range(GB1inputArrayLength): # Recall that GB1inputArrayLength is 81
            trainingData = mask1LetterList[node]  # This is the new input from the masking field results
            inputDataList.append(trainingData)
        for node in range(GB1inputArrayLength): # Second set of 81 nodes
            trainingData = mask2LetterList[node]  # This is the second set of input from the masking field results
            inputDataList.append(trainingData)          


# Fill the second part of the training data list with the outputs from GB1          
        for node in range(GB1outputArrayLength): 
            trainingData = GB1outputArray[node]  
            inputDataList.append(trainingData)


# Create an input array with both the original training data and the outputs from GB1
        for node in range(inputArrayLength): 
            inputDataArray[node] = inputDataList[node]            

                                
        letterNum = trainingDataList[2] +1
        letterChar = trainingDataList[3]  
        print ' '
        print '  Data Set Number', selectedTrainingDataSet, ' for letter ', letterChar, ' with letter number ', letterNum 

        if trainingDataList[4]==0:
            hiddenArray0 = ComputeSingleFeedforwardPassFirstStep0 (alpha, inputDataArray, wWeightArray0, biasHiddenWeightArray0)
        elif trainingDataList[4]==1:
            hiddenArray1 = ComputeSingleFeedforwardPassFirstStep1 (alpha, inputDataArray, wWeightArray1, biasHiddenWeightArray1)
        elif trainingDataList[4]==2:
            hiddenArray2 = ComputeSingleFeedforwardPassFirstStep2 (alpha, inputDataArray, wWeightArray2, biasHiddenWeightArray2)
        elif trainingDataList[4]==3:
            hiddenArray3 = ComputeSingleFeedforwardPassFirstStep3 (alpha, inputDataArray, wWeightArray3, biasHiddenWeightArray3)
        else:
            hiddenArray4 = ComputeSingleFeedforwardPassFirstStep4 (alpha, inputDataArray, wWeightArray4, biasHiddenWeightArray4)


        if trainingDataList[4]==0:
            outputArray = ComputeSingleFeedforwardPassSecondStep0 (alpha, hiddenArray0, vWeightArray0, biasOutputWeightArray0)
        elif trainingDataList[4]==1:
            outputArray = ComputeSingleFeedforwardPassSecondStep1 (alpha, hiddenArray1, vWeightArray1, biasOutputWeightArray1)            
        elif trainingDataList[4]==2:
            outputArray = ComputeSingleFeedforwardPassSecondStep2 (alpha, hiddenArray2, vWeightArray2, biasOutputWeightArray2)            
        elif trainingDataList[4]==3:
            outputArray = ComputeSingleFeedforwardPassSecondStep3 (alpha, hiddenArray3, vWeightArray3, biasOutputWeightArray3)        
        else:
            outputArray = ComputeSingleFeedforwardPassSecondStep4 (alpha, hiddenArray4, vWeightArray4, biasOutputWeightArray4)                    
                
        print ' '
        print ' The output node activations are:'
        print outputArray   

        desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
        desiredClass = trainingDataList[2]                 # identify the desired class
        desiredOutputArray[desiredClass] = 1                # set the desired output for that class to 1
     
        print ' '
        print ' The desired output array values are: '
        print desiredOutputArray  
       
                        
# Determine the error between actual and desired outputs

# Initialize the error array
        errorArray = np.zeros(outputArrayLength) 
    
        newSSE = 0.0
        for node in range(outputArrayLength):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

        newTotalSSE += newSSE
        misclassification += round(max(abs(errorArray)),0)
        
#        print ' '
#        print ' The error values are:'
#        print errorArray 
        
        print ' '
        print' The max error value is:', round(max(abs(errorArray)),0) 
        
# Print the Summed Squared Error  
        print 'New SSE = %.6f' % newSSE 
        print 'New Total SSE = %.6f' % newTotalSSE 
        print 'New Total Misclassificaiton = %.6f' % misclassification 
        
        selectedTrainingDataSet = selectedTrainingDataSet +1 
        

                        


####################################################################################################
#**************************************************************************************************#
####################################################################################################
#
#   Backpropgation Section
#
####################################################################################################
#**************************************************************************************************#
####################################################################################################

   
            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the hidden-to-output connection weights
#
####################################################################################################
####################################################################################################


def backpropagateOutputToHidden0 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray0, vWeightArray0):

# The first step here applies a backpropagation-based weight change to the hidden-to-output wts v. 
# Core equation for the first part of backpropagation: 
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# -- Hidden(h) = the output of hidden node h. 

# We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight v. 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com


# Unpack array lengths
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]

    transferFuncDerivArray0 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray0[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 


# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in the equations for the deltas in the connection weights    
                        
    deltaVWtArray0 = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray0 = np.zeros((outputArrayLength, hiddenArrayLength)) # initialize an array for the new hidden weights
        
    for row in range(outputArrayLength):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes,
        #    and the columns correspond to the number of hidden nodes,
        #    which can be multiplied by the hidden node array (expressed as a column).
        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt0 = -errorArray[row]*transferFuncDerivArray0[row]*hiddenArray0[col]
            deltaVWtArray0[row,col] = -eta*partialSSE_w_V_Wt0
            newVWeightArray0[row,col] = vWeightArray0[row,col] + deltaVWtArray0[row,col]                                                                                       
                                                                                                                                                                                                                                                                           
    return (newVWeightArray0);     

def backpropagateOutputToHidden1 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray1, vWeightArray1):


# Unpack array lengths
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]

    transferFuncDerivArray1 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray1[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 


# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in the equations for the deltas in the connection weights    
                        
    deltaVWtArray1 = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray1 = np.zeros((outputArrayLength, hiddenArrayLength)) # initialize an array for the new hidden weights
        
    for row in range(outputArrayLength):  #  Number of rows in weightMatrix

        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt1 = -errorArray[row]*transferFuncDerivArray1[row]*hiddenArray1[col]
            deltaVWtArray1[row,col] = -eta*partialSSE_w_V_Wt1
            newVWeightArray1[row,col] = vWeightArray1[row,col] + deltaVWtArray1[row,col]                                                                                     
                                                                                                                                                                                                                                                                           
    return (newVWeightArray1);  
    
def backpropagateOutputToHidden2 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray2, vWeightArray2):

# Unpack array lengths
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]

    transferFuncDerivArray2 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray2[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 

    deltaVWtArray2 = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray2 = np.zeros((outputArrayLength, hiddenArrayLength)) # initialize an array for the new hidden weights
        
    for row in range(outputArrayLength):  #  Number of rows in weightMatrix

        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt2 = -errorArray[row]*transferFuncDerivArray2[row]*hiddenArray2[col]
            deltaVWtArray2[row,col] = -eta*partialSSE_w_V_Wt2
            newVWeightArray2[row,col] = vWeightArray2[row,col] + deltaVWtArray2[row,col]                                                                                       
                                                                                                                                                                                                                                                                              
    return (newVWeightArray2);  
    
def backpropagateOutputToHidden3 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray3, vWeightArray3):

# Unpack array lengths
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]

    transferFuncDerivArray3 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray3[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
                         
    deltaVWtArray3 = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray3 = np.zeros((outputArrayLength, hiddenArrayLength)) # initialize an array for the new hidden weights
        
    for row in range(outputArrayLength):  #  Number of rows in weightMatrix
        
        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt3 = -errorArray[row]*transferFuncDerivArray3[row]*hiddenArray3[col]
            deltaVWtArray3[row,col] = -eta*partialSSE_w_V_Wt3
            newVWeightArray3[row,col] = vWeightArray3[row,col] + deltaVWtArray3[row,col]                                                                                                                                                  
                                                                                                                                                                                                            
    return (newVWeightArray3);  
    
def backpropagateOutputToHidden4 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray4, vWeightArray4):

# Unpack array lengths
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]

    transferFuncDerivArray4 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray4[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 
    deltaVWtArray4 = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray4 = np.zeros((outputArrayLength, hiddenArrayLength)) # initialize an array for the new hidden weights
        
    for row in range(outputArrayLength):  #  Number of rows in weightMatrix

        for col in range(hiddenArrayLength):  # number of columns in weightMatrix 
            partialSSE_w_V_Wt4 = -errorArray[row]*transferFuncDerivArray4[row]*hiddenArray4[col]
            deltaVWtArray4[row,col] = -eta*partialSSE_w_V_Wt4
            newVWeightArray4[row,col] = vWeightArray4[row,col] + deltaVWtArray4[row,col]                                                                                                                                                    
                                                                                                                                                                                                            
    return (newVWeightArray4);  
            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-output connection weights
#
####################################################################################################
####################################################################################################


def backpropagateBiasOutputWeights0 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray0):

# The first step here applies a backpropagation-based weight change to the hidden-to-output wts v. 
# Core equation for the first part of backpropagation: 
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# -- Hidden(h) = the output of hidden node h. 

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight biasOutput(o). 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com


# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations 

# The equation for the actual dependence of the Summed Squared Error on a given bias-to-output 
#   weight biasOutput(o) is:
#   partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
# The transfer function derivative (transFuncDeriv) returned from computeTransferFnctnDeriv is given as:
#   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
# Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
#   partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
#   The parameter alpha is included in transFuncDeriv


# Unpack the output array length
    outputArrayLength = arraySizeList [5]

    deltaBiasOutputArray0 = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray0 = np.zeros(outputArrayLength) # initialize an array for the new output bias weights
    transferFuncDerivArray0 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray0[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 

    for node in range(outputArrayLength):  #  Number of nodes in output array (same as number of output bias nodes)    
        partialSSE_w_BiasOutput0 = -errorArray[node]*transferFuncDerivArray0[node]
        deltaBiasOutputArray0[node] = -eta*partialSSE_w_BiasOutput0  
        newBiasOutputWeightArray0[node] =  biasOutputWeightArray0[node] + deltaBiasOutputArray0[node]           
   
#    print ' '
#    print ' The previous biases for the output nodes are: '
#    print biasOutputWeightArray
#    print ' '
#    print ' The new biases for the output nodes are: '
#    print newBiasOutputWeightArray
                                                                                                                                                
    return (newBiasOutputWeightArray0);     

def backpropagateBiasOutputWeights1 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray1):

# Unpack the output array length
    outputArrayLength = arraySizeList [5]

    deltaBiasOutputArray1 = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray1 = np.zeros(outputArrayLength) # initialize an array for the new output bias weights
    transferFuncDerivArray1 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray1[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 

    for node in range(outputArrayLength):  #  Number of nodes in output array (same as number of output bias nodes)    
        partialSSE_w_BiasOutput1 = -errorArray[node]*transferFuncDerivArray1[node]
        deltaBiasOutputArray1[node] = -eta*partialSSE_w_BiasOutput1  
        newBiasOutputWeightArray1[node] =  biasOutputWeightArray1[node] + deltaBiasOutputArray1[node]           
                                                                                                                                                
    return (newBiasOutputWeightArray1);  
    
def backpropagateBiasOutputWeights2 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray2):

# Unpack the output array length
    outputArrayLength = arraySizeList [5]

    deltaBiasOutputArray2 = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray2 = np.zeros(outputArrayLength) # initialize an array for the new output bias weights
    transferFuncDerivArray2 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray2[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 

    for node in range(outputArrayLength):  #  Number of nodes in output array (same as number of output bias nodes)    
        partialSSE_w_BiasOutput2 = -errorArray[node]*transferFuncDerivArray2[node]
        deltaBiasOutputArray2[node] = -eta*partialSSE_w_BiasOutput2 
        newBiasOutputWeightArray2[node] =  biasOutputWeightArray2[node] + deltaBiasOutputArray2[node]           
                                                                                                                                                
    return (newBiasOutputWeightArray2);  
    
def backpropagateBiasOutputWeights3 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray3):

# Unpack the output array length
    outputArrayLength = arraySizeList [5]

    deltaBiasOutputArray3 = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray3 = np.zeros(outputArrayLength) # initialize an array for the new output bias weights
    transferFuncDerivArray3 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray3[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 

    for node in range(outputArrayLength):  #  Number of nodes in output array (same as number of output bias nodes)    
        partialSSE_w_BiasOutput3 = -errorArray[node]*transferFuncDerivArray3[node]
        deltaBiasOutputArray3[node] = -eta*partialSSE_w_BiasOutput3 
        newBiasOutputWeightArray3[node] =  biasOutputWeightArray3[node] + deltaBiasOutputArray3[node]           
                                                                                                                                                
    return (newBiasOutputWeightArray3);  
    
def backpropagateBiasOutputWeights4 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray4):

# Unpack the output array length
    outputArrayLength = arraySizeList [5]

    deltaBiasOutputArray4 = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray4 = np.zeros(outputArrayLength) # initialize an array for the new output bias weights
    transferFuncDerivArray4 = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray4[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 

    for node in range(outputArrayLength):  #  Number of nodes in output array (same as number of output bias nodes)    
        partialSSE_w_BiasOutput4 = -errorArray[node]*transferFuncDerivArray4[node]
        deltaBiasOutputArray4[node] = -eta*partialSSE_w_BiasOutput4  
        newBiasOutputWeightArray4[node] =  biasOutputWeightArray4[node] + deltaBiasOutputArray4[node]           
                                                                                                                                                
    return (newBiasOutputWeightArray4);  

####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the input-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def backpropagateHiddenToInput0 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray0,
    inputArray, vWeightArray0, wWeightArray0, biasHiddenWeightArray0, biasOutputWeightArray0):

# The first step here applies a backpropagation-based weight change to the input-to-hidden wts w. 
# Core equation for the second part of backpropagation: 
# d(SSE)/dw(i,h) = -eta*alpha*F(h)(1-F(h))*Input(i)*sum(v(h,o)*Error(o))
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- w(i,h) is the connection weight w between the input node i and the hidden node h
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1 
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# ---- NOTE: in this second step, the transfer function is applied to the output of the hidden node,
# ------ so that F = F(h)
# -- Hidden(h) = the output of hidden node h (used in computing the derivative of the transfer function). 
# -- Input(i) = the input at node i.

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# Unpack the errorList and the vWeightArray

# We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight w. 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight
 
# For the second step in backpropagation (computing deltas on the input-to-hidden weights)
#   we need the transfer function derivative is applied to the output at the hidden node        

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations       
    transferFuncDerivHiddenArray0 = np.zeros(hiddenArrayLength)    # initialize an array for the transfer function deriv 
      
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray0[node]=computeTransferFnctnDeriv(hiddenArray0[node], alpha)
        
    errorTimesTFuncDerivOutputArray0 = np.zeros(outputArrayLength) # initialize array
    transferFuncDerivOutputArray0    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray0              = np.zeros(hiddenArrayLength) # initialize array
      
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray0[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray0[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray0[outputNode]
        
    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray0[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray0[hiddenNode] = weightedErrorArray0[hiddenNode] \
            + vWeightArray0[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray0[outputNode]
             
    deltaWWtArray0 = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray0 = np.zeros((hiddenArrayLength, inputArrayLength)) # initialize an array for the new input-to-hidden weights
        
    for row in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts0 = -transferFuncDerivHiddenArray0[row]*inputArray[col]*weightedErrorArray0[row]
            deltaWWtArray0[row,col] = -eta*partialSSE_w_W_Wts0
            newWWeightArray0[row,col] = wWeightArray0[row,col] + deltaWWtArray0[row,col]                                                                                           
                                                                    
    return (newWWeightArray0);     
    

def backpropagateHiddenToInput1 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray1,
    inputArray, vWeightArray1, wWeightArray1, biasHiddenWeightArray1, biasOutputWeightArray1):

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations       
    transferFuncDerivHiddenArray1 = np.zeros(hiddenArrayLength)    # initialize an array for the transfer function deriv 
      
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray1[node]=computeTransferFnctnDeriv(hiddenArray1[node], alpha)
        
    errorTimesTFuncDerivOutputArray1 = np.zeros(outputArrayLength) # initialize array
    transferFuncDerivOutputArray1    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray1              = np.zeros(hiddenArrayLength) # initialize array
      
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray1[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray1[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray1[outputNode]
        
    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray1[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray1[hiddenNode] = weightedErrorArray1[hiddenNode] \
            + vWeightArray1[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray1[outputNode]
             
    deltaWWtArray1 = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray1 = np.zeros((hiddenArrayLength, inputArrayLength)) # initialize an array for the new input-to-hidden weights
        
    for row in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts1 = -transferFuncDerivHiddenArray1[row]*inputArray[col]*weightedErrorArray1[row]
            deltaWWtArray1[row,col] = -eta*partialSSE_w_W_Wts1
            newWWeightArray1[row,col] = wWeightArray1[row,col] + deltaWWtArray1[row,col]                                                                                     
                                                                   
    return (newWWeightArray1);        
    
def backpropagateHiddenToInput2 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray2,
    inputArray, vWeightArray2, wWeightArray2, biasHiddenWeightArray2, biasOutputWeightArray2):

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations       
    transferFuncDerivHiddenArray2 = np.zeros(hiddenArrayLength)    # initialize an array for the transfer function deriv 
      
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray2[node]=computeTransferFnctnDeriv(hiddenArray2[node], alpha)
        
    errorTimesTFuncDerivOutputArray2 = np.zeros(outputArrayLength) # initialize array
    transferFuncDerivOutputArray2    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray2              = np.zeros(hiddenArrayLength) # initialize array
      
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray2[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray2[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray2[outputNode]
        
    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray2[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray2[hiddenNode] = weightedErrorArray2[hiddenNode] \
            + vWeightArray2[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray2[outputNode]
             
    deltaWWtArray2 = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray2 = np.zeros((hiddenArrayLength, inputArrayLength)) # initialize an array for the new input-to-hidden weights
        
    for row in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts2 = -transferFuncDerivHiddenArray2[row]*inputArray[col]*weightedErrorArray2[row]
            deltaWWtArray2[row,col] = -eta*partialSSE_w_W_Wts2
            newWWeightArray2[row,col] = wWeightArray2[row,col] + deltaWWtArray2[row,col]                                                                                     
                                                                    
    return (newWWeightArray2);       
    
def backpropagateHiddenToInput3 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray3,
    inputArray, vWeightArray3, wWeightArray3, biasHiddenWeightArray3, biasOutputWeightArray3):

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations       
    transferFuncDerivHiddenArray3 = np.zeros(hiddenArrayLength)    # initialize an array for the transfer function deriv 
      
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray3[node]=computeTransferFnctnDeriv(hiddenArray3[node], alpha)
        
    errorTimesTFuncDerivOutputArray3 = np.zeros(outputArrayLength) # initialize array
    transferFuncDerivOutputArray3    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray3              = np.zeros(hiddenArrayLength) # initialize array
      
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray3[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray3[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray3[outputNode]
        
    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray3[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray3[hiddenNode] = weightedErrorArray3[hiddenNode] \
            + vWeightArray3[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray3[outputNode]
             
    deltaWWtArray3 = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray3 = np.zeros((hiddenArrayLength, inputArrayLength)) # initialize an array for the new input-to-hidden weights
        
    for row in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts3 = -transferFuncDerivHiddenArray3[row]*inputArray[col]*weightedErrorArray3[row]
            deltaWWtArray3[row,col] = -eta*partialSSE_w_W_Wts3
            newWWeightArray3[row,col] = wWeightArray3[row,col] + deltaWWtArray3[row,col]                                                                                     
                                                                    
    return (newWWeightArray3);       
    
def backpropagateHiddenToInput4 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray4,
    inputArray, vWeightArray4, wWeightArray4, biasHiddenWeightArray4, biasOutputWeightArray4):

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations       
    transferFuncDerivHiddenArray4 = np.zeros(hiddenArrayLength)    # initialize an array for the transfer function deriv 
      
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray4[node]=computeTransferFnctnDeriv(hiddenArray4[node], alpha)
        
    errorTimesTFuncDerivOutputArray4 = np.zeros(outputArrayLength) # initialize array
    transferFuncDerivOutputArray4    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray4              = np.zeros(hiddenArrayLength) # initialize array
      
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray4[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray4[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray4[outputNode]
        
    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray4[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray4[hiddenNode] = weightedErrorArray4[hiddenNode] \
            + vWeightArray4[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray4[outputNode]
             
    deltaWWtArray4 = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray4 = np.zeros((hiddenArrayLength, inputArrayLength)) # initialize an array for the new input-to-hidden weights
        
    for row in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts4 = -transferFuncDerivHiddenArray4[row]*inputArray[col]*weightedErrorArray4[row]
            deltaWWtArray4[row,col] = -eta*partialSSE_w_W_Wts4
            newWWeightArray4[row,col] = wWeightArray4[row,col] + deltaWWtArray4[row,col]                                                                                     
                                                                   
    return (newWWeightArray4);                
                                    
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def backpropagateBiasHiddenWeights0 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray0,
    inputArray, vWeightArray0, wWeightArray0, biasHiddenWeightArray0, biasOutputWeightArray0):

# The first step here applies a backpropagation-based weight change to the hidden-to-output wts v. 
# Core equation for the first part of backpropagation: 
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# -- Hidden(h) = the output of hidden node h. 

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight biasOutput(o). 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com


# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          
   
# Compute the transfer function derivatives as a function of the output nodes.
# Note: As this is being done after the call to the backpropagation on the hidden-to-output weights,
#   the transfer function derivative computed there could have been used here; the calculations are
#   being redone here only to maintain module independence              

    errorTimesTFuncDerivOutputArray0 = np.zeros(outputArrayLength) # initialize array    
    transferFuncDerivOutputArray0    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray0              = np.zeros(hiddenArrayLength) # initialize array    

    transferFuncDerivHiddenArray0 = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv 
    partialSSE_w_BiasHidden0      = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray0         = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray0     = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights
          
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray0[node]=computeTransferFnctnDeriv(hiddenArray0[node], alpha)      
                  
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray0[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha) 
        errorTimesTFuncDerivOutputArray0[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray0[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray0[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray0[hiddenNode] = weightedErrorArray0[hiddenNode]
            + vWeightArray0[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray0[outputNode]
            
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations 


# ===>>> AJM needs to double-check these equations in the comments area
# ===>>> The code should be fine. 
# The equation for the actual dependence of the Summed Squared Error on a given bias-to-output 
#   weight biasOutput(o) is:
#   partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
# The transfer function derivative (transFuncDeriv) returned from computeTransferFnctnDeriv is given as:
#   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
# Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
#   partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
#   The parameter alpha is included in transFuncDeriv

    for hiddenNode in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix           
        partialSSE_w_BiasHidden0[hiddenNode] = -transferFuncDerivHiddenArray0[hiddenNode]*weightedErrorArray0[hiddenNode]
        deltaBiasHiddenArray0[hiddenNode] = -eta*partialSSE_w_BiasHidden0[hiddenNode]
        newBiasHiddenWeightArray0[hiddenNode] = biasHiddenWeightArray0[hiddenNode] + deltaBiasHiddenArray0[hiddenNode]                                                                                                                                                                                                                                                         
  
                                                                                                                                            
    return (newBiasHiddenWeightArray0); 


def backpropagateBiasHiddenWeights1 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray1,
    inputArray, vWeightArray1, wWeightArray1, biasHiddenWeightArray1, biasOutputWeightArray1):

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          
    errorTimesTFuncDerivOutputArray1 = np.zeros(outputArrayLength) # initialize array    
    transferFuncDerivOutputArray1    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray1              = np.zeros(hiddenArrayLength) # initialize array    

    transferFuncDerivHiddenArray1 = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv 
    partialSSE_w_BiasHidden1      = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray1         = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray1     = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights
          
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray1[node]=computeTransferFnctnDeriv(hiddenArray1[node], alpha)      
                  
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray1[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha) 
        errorTimesTFuncDerivOutputArray1[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray1[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray1[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray1[hiddenNode] = weightedErrorArray1[hiddenNode]
            + vWeightArray1[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray1[outputNode]

    for hiddenNode in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix           
        partialSSE_w_BiasHidden1[hiddenNode] = -transferFuncDerivHiddenArray1[hiddenNode]*weightedErrorArray1[hiddenNode]
        deltaBiasHiddenArray1[hiddenNode] = -eta*partialSSE_w_BiasHidden1[hiddenNode]
        newBiasHiddenWeightArray1[hiddenNode] = biasHiddenWeightArray1[hiddenNode] + deltaBiasHiddenArray1[hiddenNode]                                                                                                                                                                                                                                                         
  
                                                                                                                                            
    return (newBiasHiddenWeightArray1);                               
    
def backpropagateBiasHiddenWeights2 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray2,
    inputArray, vWeightArray2, wWeightArray2, biasHiddenWeightArray2, biasOutputWeightArray2):

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              

    errorTimesTFuncDerivOutputArray2 = np.zeros(outputArrayLength) # initialize array    
    transferFuncDerivOutputArray2    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray2              = np.zeros(hiddenArrayLength) # initialize array    

    transferFuncDerivHiddenArray2 = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv 
    partialSSE_w_BiasHidden2      = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray2         = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray2     = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights
          
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray2[node]=computeTransferFnctnDeriv(hiddenArray2[node], alpha)      
                  
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray2[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha) 
        errorTimesTFuncDerivOutputArray2[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray2[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray2[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray2[hiddenNode] = weightedErrorArray2[hiddenNode]
            + vWeightArray2[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray2[outputNode]

    for hiddenNode in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix           
        partialSSE_w_BiasHidden2[hiddenNode] = -transferFuncDerivHiddenArray2[hiddenNode]*weightedErrorArray2[hiddenNode]
        deltaBiasHiddenArray2[hiddenNode] = -eta*partialSSE_w_BiasHidden2[hiddenNode]
        newBiasHiddenWeightArray2[hiddenNode] = biasHiddenWeightArray2[hiddenNode] + deltaBiasHiddenArray2[hiddenNode]                                                                                                                                                                                                                                                         
  
                                                                                                                                            
    return (newBiasHiddenWeightArray2);                               
    
def backpropagateBiasHiddenWeights3 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray3,
    inputArray, vWeightArray3, wWeightArray3, biasHiddenWeightArray3, biasOutputWeightArray3):

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]                

    errorTimesTFuncDerivOutputArray3 = np.zeros(outputArrayLength) # initialize array    
    transferFuncDerivOutputArray3    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray3              = np.zeros(hiddenArrayLength) # initialize array    

    transferFuncDerivHiddenArray3 = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv 
    partialSSE_w_BiasHidden3      = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray3         = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray3     = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights
          
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray3[node]=computeTransferFnctnDeriv(hiddenArray3[node], alpha)      
                  
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray3[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha) 
        errorTimesTFuncDerivOutputArray3[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray3[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray3[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray3[hiddenNode] = weightedErrorArray3[hiddenNode]
            + vWeightArray3[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray3[outputNode]

    for hiddenNode in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix           
        partialSSE_w_BiasHidden3[hiddenNode] = -transferFuncDerivHiddenArray3[hiddenNode]*weightedErrorArray3[hiddenNode]
        deltaBiasHiddenArray3[hiddenNode] = -eta*partialSSE_w_BiasHidden3[hiddenNode]
        newBiasHiddenWeightArray3[hiddenNode] = biasHiddenWeightArray3[hiddenNode] + deltaBiasHiddenArray3[hiddenNode]                                                                                                                                                                                                                                                         
  
                                                                                                                                            
    return (newBiasHiddenWeightArray3);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

def backpropagateBiasHiddenWeights4 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray4,
    inputArray, vWeightArray4, wWeightArray4, biasHiddenWeightArray4, biasOutputWeightArray4):

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              

    errorTimesTFuncDerivOutputArray4 = np.zeros(outputArrayLength) # initialize array    
    transferFuncDerivOutputArray4    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray4              = np.zeros(hiddenArrayLength) # initialize array    

    transferFuncDerivHiddenArray4 = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv 
    partialSSE_w_BiasHidden4      = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray4         = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray4     = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights
          
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray4[node]=computeTransferFnctnDeriv(hiddenArray4[node], alpha)      
                  
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray4[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha) 
        errorTimesTFuncDerivOutputArray4[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray4[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray4[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray4[hiddenNode] = weightedErrorArray4[hiddenNode]
            + vWeightArray4[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray4[outputNode]

    for hiddenNode in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix           
        partialSSE_w_BiasHidden4[hiddenNode] = -transferFuncDerivHiddenArray4[hiddenNode]*weightedErrorArray4[hiddenNode]
        deltaBiasHiddenArray4[hiddenNode] = -eta*partialSSE_w_BiasHidden4[hiddenNode]
        newBiasHiddenWeightArray4[hiddenNode] = biasHiddenWeightArray4[hiddenNode] + deltaBiasHiddenArray4[hiddenNode]                                                                                                                                                                                                                                                         
                                                                                                                                              
    return (newBiasHiddenWeightArray4);                               


####################################################################################################
####################################################################################################
#
# The following modules expand the boundaries around a chosen letter, and apply a masking filter to 
#   that expanded letter. The result is an array (9x9 in this case) of units, with activation values
#   where 0 <= v <= 1.  
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
# 
####################################################################################################
####################################################################################################
#
# Function to expand the grid containing a letter by one pixel in each direction
#
####################################################################################################
####################################################################################################

def expandLetterBoundaries (ExpandTrainingDataList):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    pixelArray = ExpandTrainingDataList


    expandedLetterArray = np.zeros(shape=(eGH,eGW)) 

    iterAcrossRow = 0
    iterOverAllRows = 0



# For logical completeness: The first element of each row in the expanded letter is set to zero
    iterAcrossRow = 0

    while iterAcrossRow < eGW:
        expandedLetterArray[iterOverAllRows,iterAcrossRow] = 0  
#        print iterAcrossRow, expandedLetterArray[iterOverAllRows,iterAcrossRow] 
        iterAcrossRow = iterAcrossRow + 1


# Fill in the elements of the expandedLetterArray; rows 1 .. eGH-1
  
    rowVal = 1
    while rowVal <eGH-2:

# For the next gridWidth elements in the row, in the expanded letter is set to zero       
        iterAcrossRow = 0
        expandedLetterArray[iterOverAllRows,iterAcrossRow] = 0
   
        iterAcrossRow = 1       
        while iterAcrossRow < eGW-2:
            expandedLetterArray[rowVal,iterAcrossRow] = 0
            #Note: We start counting in the pixelArray at iterAcrossRow-1, because that array 
            #      starts at with the first element at position '0'
            #      and iterAcrossRow is one count beyond that 
            
            if pixelArray[iterAcrossRow-2+(rowVal-2)*gridWidth] > 0.9: 
                expandedLetterArray[rowVal,iterAcrossRow] = 1

            iterAcrossRow = iterAcrossRow +1

        iterAcrossRow = 0  #re-initialize iteration count  
        rowVal = rowVal +1

        # For logical completeness: The last element of each row in the expanded letter is set to zero
        # Note: The last element in the row is at position eGW-1, as the row count starts with 0
    rowVal = eGH-1
    iterAcrossRow = 0
    while iterAcrossRow < eGW-1:
        expandedLetterArray[rowVal,iterAcrossRow] = 0  

        iterAcrossRow = iterAcrossRow + 1      
         
    return expandedLetterArray




####################################################################################################
####################################################################################################
#
# Function to return the letterArray after mask1 has been applied to it
#
####################################################################################################
####################################################################################################


def maskLetterFunc(expandedLetterArray, mask):

    
    maskLetterArray = np.zeros(shape=(gridHeight,gridWidth))
    
   
    rowVal = 1
    colVal = 1

    
    while rowVal <gridHeight+1: 
#        print ' '
#        print ' for expanded letter row = ', rowVal
        arrayRow = rowVal - 1 
#        print '   for the masked result letter row = ', arrayRow
#        print '   for the maasked result letter column:'
        while colVal <gridWidth+1:           
            e0 =  expandedLetterArray[rowVal-1, colVal-1]
            e1 =  expandedLetterArray[rowVal-1, colVal]
            e2 =  expandedLetterArray[rowVal-1, colVal+1]   
            e3 =  expandedLetterArray[rowVal, colVal-1]
            e4 =  expandedLetterArray[rowVal, colVal]
            e5 =  expandedLetterArray[rowVal, colVal+1]   
            e6 =  expandedLetterArray[rowVal+1, colVal-1]
            e7 =  expandedLetterArray[rowVal+1, colVal]
            e8 =  expandedLetterArray[rowVal+1, colVal+1]               
              
            maskArrayVal    =  (e0*mask[0] + e1*mask[1] + e2*mask[2] + 
                                e3*mask[3] + e4*mask[4] + e5*mask[5] + 
                                e6*mask[6] + e7*mask[7] + e8*mask[8] ) / 3.0                        
                         
            arrayCol = colVal - 1

            maskLetterArray[arrayRow,arrayCol] = maskArrayVal 
#            print ' col = ', arrayCol, 'val = %.3f' % mask1ArrayVal                                                                                                                                          
            colVal = colVal + 1

        rowVal = rowVal + 1
        colVal = 1              
                                                               
    return maskLetterArray 
    



####################################################################################################
####################################################################################################
#
# Procedure to convert the 2x2 array produced by maskLetter into a list and return the list 
#
####################################################################################################
####################################################################################################

def convertArrayToList(maskLetterArray):

    maskLetterList = list()

    for row in range(gridHeight):  #  Number of rows in a masked input grid
        for col in range(gridWidth):  # number of columns in a masked input grid
            localGridElement = maskLetterArray[row,col] 
            maskLetterList.append(localGridElement)   

    return (maskLetterList)    
            

####################################################################################################
####################################################################################################
#
# The following are a series of functions to access the data files and convert the retrieved data
#   from lists into arrays
#
####################################################################################################
####################################################################################################

####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def readGB1wWeightFile (): 

    GB1wWeightList = list()
    with open('C:\Users\chauh\Documents\Susan Data\GB1wWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:

            colnum = 0
            theRow = row
            for col in row:

                data = float(theRow[colnum])

            GB1wWeightList.append(data)
       
    return GB1wWeightList                                                  


####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def readGB1vWeightFile (): 

    GB1vWeightList = list()
    with open('C:\Users\chauh\Documents\Susan Data\GB1vWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:

            colnum = 0
            theRow = row
            for col in row:

                data = float(theRow[colnum])

            GB1vWeightList.append(data)
      
    return GB1vWeightList                                                  

####################################################################################################
#**************************************************************************************************#
####################################################################################################

def reconstructGB1wWeightArray (GB1wWeightList):

    numUpperNodes = GB1hiddenArrayLength
    numLowerNodes = GB1inputArrayLength 
    
    GB1wWeightArray = np.zeros((numUpperNodes,numLowerNodes))    # initialize the weight matrix with 0's     
 
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.

        for col in range(numLowerNodes):  # number of columns in matrix 2
            localPosition = row*numLowerNodes + col            
            localWeight = GB1wWeightList[localPosition]

            GB1wWeightArray[row,col] = localWeight

                                                     
    return GB1wWeightArray  



####################################################################################################
#**************************************************************************************************#
####################################################################################################

def reconstructGB1vWeightArray (GB1vWeightList):

    numUpperNodes = GB1outputArrayLength
    numLowerNodes = GB1hiddenArrayLength 
    
    GB1vWeightArray = np.zeros((numUpperNodes,numLowerNodes))    # iniitalize the weight matrix with 0's     
  
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For a hidden-to-output weight matrix, the rows correspond to the number of output nodes
        #    and the columns correspond to the number of hidden nodes.
        #    This creates an OxH matrix, which can be multiplied by the hidden nodes matrix (expressed as a column)

        for col in range(numLowerNodes):  # number of columns in matrix 2
            localPosition = row*numLowerNodes + col
            localWeight = GB1vWeightList[localPosition]
            GB1vWeightArray[row,col] = localWeight
                                                     
    return GB1vWeightArray  
    


####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def readGB1wBiasWeightFile (): 

    GB1wBiasWeightList = list()
    with open('C:\Users\chauh\Documents\Susan Data\GB1wBiasWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:

            colnum = 0
            theRow = row
            for col in row:

                data = float(theRow[colnum])

            GB1wBiasWeightList.append(data)
     
    return GB1wBiasWeightList                                                  


####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def readGB1vBiasWeightFile (): 

    GB1vBiasWeightList = list()
    with open('C:\Users\chauh\Documents\Susan Data\GB1vBiasWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:

            colnum = 0
            theRow = row
            for col in row:

                data = float(theRow[colnum])

            GB1vBiasWeightList.append(data)
      
    return GB1vBiasWeightList                                                  

####################################################################################################
#**************************************************************************************************#
####################################################################################################

def reconstructGB1wBiasWeightArray (GB1wBiasWeightList):

    GB1wBiasWeightArray = np.zeros(GB1hiddenArrayLength)    # initialize the weight matrix with 0's     

    for node in range(GB1hiddenArrayLength):  #  Number of hidden bias nodes          
            localWeight = GB1wBiasWeightList[node]
            GB1wBiasWeightArray[node] = localWeight
               
    return GB1wBiasWeightArray  



####################################################################################################
#**************************************************************************************************#
####################################################################################################

def reconstructGB1vBiasWeightArray (GB1vBiasWeightList):
    
    GB1vBiasWeightArray = np.zeros(GB1outputArrayLength)    # iniitalize the weight matrix with 0's     
  
    for node in range(GB1outputArrayLength):  #  Number of output bias nodes
            localWeight = GB1vBiasWeightList[node]
            GB1vBiasWeightArray[node] = localWeight
                                                     
    return GB1vBiasWeightArray  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
####################################################################################################
#**************************************************************************************************#
####################################################################################################    
        
            
                    
            
####################################################################################################
#**************************************************************************************************#
####################################################################################################
#
# The MAIN module comprising of calls to:
#   (1) Welcome
#   (2) Obtain neural network size specifications for a three-layer network consisting of:
#       - Input layer
#       - Hidden layer
#       - Output layer (all the sizes are currently hard-coded to two nodes per layer right now)
#   (3) Initialize connection weight values
#       - w: Input-to-Hidden nodes
#       - v: Hidden-to-Output nodes
#   (4) Compute a feedforward pass in two steps
#       - Randomly select a single training data set
#       - Input-to-Hidden
#       - Hidden-to-Output
#       - Compute the error array
#       - Compute the new Summed Squared Error (SSE)
#   (5) Perform a single backpropagation training pass

# (not yet complete; needs updating)
#
####################################################################################################
#**************************************************************************************************#
####################################################################################################


def main():

# Define the global variables        
    global inputArrayLength
    global hiddenArrayLength
    global outputArrayLength
    global GB1inputArrayLength
    global GB1hiddenArrayLength
    global GB1outputArrayLength    
    global gridWidth
    global gridHeight
    global eGH # expandedGridHeight, defined in function expandLetterBoundaries 
    global eGW # expandedGridWidth defined in function expandLetterBoundaries 
    global mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10, mask11, mask12    

####################################################################################################
# Obtain unit array size in terms of array_length (M) and layers (N)
####################################################################################################                

# This calls the procedure 'welcome,' which just prints out a welcoming message. 
# All procedures need an argument list. 
# This procedure has a list, but it is an empty list; welcome().

    welcome()

    
# Right now, for simplicity, we're going to hard-code the numbers of layers that we have in our 
#   multilayer Perceptron (MLP) neural network. 
# We will have an input layer (I), an output layer (O), and a single hidden layer (H). 

# Define the variable arraySizeList, which is a list. It is initially an empty list. 
# Its purpose is to store the size of the array.

    arraySizeList = list() # empty list

# Obtain the actual sizes for each layer of the network       
    arraySizeList = obtainNeuralNetworkSizeSpecs ()
    
# Unpack the list; ascribe the various elements of the list to the sizes of different network layers
# Note: A word on Python encoding ... the actually length of the array, in each of these three cases, 
#       will be xArrayLength. For example, the inputArrayLength for the 9x9 pixel array is 81. 
#       These values are passed to various procedures. They start filling in actual array values,
#       where the array values start their count at element 0. However, when filling them in using a
#       "for node in range[limit]" statement, the "for" loop fills from 0 up to limit-1. Thus, the
#       original xArrayLength size is preserved.   
    GB1inputArrayLength = arraySizeList [0] 
    GB1hiddenArrayLength = arraySizeList [1] 
    GB1outputArrayLength = arraySizeList [2] 
    inputArrayLength = arraySizeList [3] 
    hiddenArrayLength = arraySizeList [4] 
    outputArrayLength = arraySizeList [5]     
    
    print ' '
    print ' inputArrayLength = ', inputArrayLength
    print ' hiddenArrayLength = ', hiddenArrayLength
    print ' outputArrayLength = ', outputArrayLength        

# Trust that the 2-D array size is the square root oft he inputArrayLength
    gridSizeFloat = (GB1inputArrayLength+1)**(1/2.0) # convert back to the total number of nodes
    gridSize = int(gridSizeFloat+0.1) # add a smidge before converting to integer

    print ' gridSize = ', gridSize

# Parameters and values for applying a masking field to the input data. 
#   Define the sizes of the letter grid; we are using a 9x9 grid for this example

    
    gridWidth = gridSize
    gridHeight = gridSize
    expandedGridHeight = gridHeight+4
    expandedGridWidth = gridWidth+4 
    eGH = expandedGridHeight
    eGW = expandedGridWidth       

    mask1 = (
    0,0,0,0,1,
    0,0,0,0,1,
    0,0,0,0,1,
    0,0,0,1,0, 
    1,1,1,0,0
    )  # O Bottom 1
    
    mask2 = (
    1,0,0,0,0,
    0,1,0,0,0,
    0,0,1,0,0,
    0,0,0,1,0, 
    0,0,0,0,1
    )  # O Diagonal \
            
    
# Parameter definitions for backpropagation, to be replaced with user inputs
    alpha = 1.0
    eta = 0.5    
    maxNumIterations = 100000    
    epsilon = 0.0000
    iteration = 0
    SSE = 0.00
    gamma = 0.1
    numTrainingDataSets = 312

                           

####################################################################################################
# 
# Grey Box 1: 
#   Read in the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
#
####################################################################################################                

# Obtain the connection weights from stored data
#
# The GB1wWeightArray is for Input-to-Hidden in Grey Box 1
# The GB1vWeightArray is for Hidden-to-Output in Grey Box 1

# Read the GB1wWeights from stored data back into this program, into a list; return the list
    GB1wWeightList = readGB1wWeightFile()
    
# Convert the GB1wWeight list back into a 2-D weight array
    GB1wWeightArray = reconstructGB1wWeightArray (GB1wWeightList) 
    
# Read the GB1vWeights from stored data back into this program, into a list; return the list
    GB1vWeightList = readGB1vWeightFile()
    
# Convert the GB1vWeight list back into a 2-D weight array
    GB1vWeightArray = reconstructGB1vWeightArray (GB1vWeightList) 
    

# Obtain the bias weights from stored data

# The GB1wBiasWeightArray is for hidden node biases in Grey Box 1
# The GB1vBiasWeightArray is for output node biases in Grey Box 1

# Read the GB1wBiasWeights from stored data back into this program, into a list; return the list
    GB1wBiasWeightList = readGB1wBiasWeightFile()
    
# Convert the GB1wBiasWeight list back into a 2-D weight array
    GB1wBiasWeightArray = reconstructGB1wBiasWeightArray (GB1wBiasWeightList) 
    
# Read the GB1vBiasWeights from stored data back into this program, into a list; return the list
    GB1vBiasWeightList = readGB1vBiasWeightFile()
    
# Convert the GB1vBiasWeight list back into a 2-D weight array
    GB1vBiasWeightArray = reconstructGB1vBiasWeightArray (GB1vBiasWeightList) 

        
####################################################################################################
# Initialize the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
####################################################################################################                

#
# The wWeightArray is for Input-to-Hidden
# The vWeightArray is for Hidden-to-Output

    wWeightArraySizeList = (inputArrayLength, hiddenArrayLength)
    vWeightArraySizeList = (hiddenArrayLength, outputArrayLength)
    biasHiddenWeightArraySize = hiddenArrayLength
    biasOutputWeightArraySize = outputArrayLength        

# The node-to-node connection weights are stored in a 2-D array
    print ' '
    print ' about to call initializeWeightArray for the w weights'
    print ' the number of lower and upper nodes is ', wWeightArraySizeList 
    wWeightArray0 = initializeWeightArray (wWeightArraySizeList)
    wWeightArray1 = initializeWeightArray (wWeightArraySizeList)
    wWeightArray2 = initializeWeightArray (wWeightArraySizeList)   
    wWeightArray3 = initializeWeightArray (wWeightArraySizeList)    
    wWeightArray4 = initializeWeightArray (wWeightArraySizeList) 
       
    print ' about to call initializeWeightArray for the v weights'
    print ' the number of lower and upper nodes is ', vWeightArraySizeList     
    vWeightArray0 = initializeWeightArray (vWeightArraySizeList)
    vWeightArray1 = initializeWeightArray (vWeightArraySizeList)
    vWeightArray2 = initializeWeightArray (vWeightArraySizeList)
    vWeightArray3 = initializeWeightArray (vWeightArraySizeList)
    vWeightArray4 = initializeWeightArray (vWeightArraySizeList)

# The bias weights are stored in a 1-D array         
    biasHiddenWeightArray0 = initializeBiasWeightArray (biasHiddenWeightArraySize)
    biasOutputWeightArray0 = initializeBiasWeightArray (biasOutputWeightArraySize) 
    biasHiddenWeightArray1 = initializeBiasWeightArray (biasHiddenWeightArraySize)
    biasOutputWeightArray1 = initializeBiasWeightArray (biasOutputWeightArraySize) 
    biasHiddenWeightArray2 = initializeBiasWeightArray (biasHiddenWeightArraySize)
    biasOutputWeightArray2 = initializeBiasWeightArray (biasOutputWeightArraySize) 
    biasHiddenWeightArray3 = initializeBiasWeightArray (biasHiddenWeightArraySize)
    biasOutputWeightArray3 = initializeBiasWeightArray (biasOutputWeightArraySize) 
    biasHiddenWeightArray4 = initializeBiasWeightArray (biasHiddenWeightArraySize)
    biasOutputWeightArray4 = initializeBiasWeightArray (biasOutputWeightArraySize) 
    
          
####################################################################################################
# Starting the backpropagation work
####################################################################################################     



# Notice in the very beginning of the program, we have 
#   np.set_printoptions(precision=4) (sets number of dec. places in print)
#     and 'np.set_printoptions(suppress=True)', which keeps it from printing in scientific format
#   Debug print: 
#    print
#    print 'The initial weights for this neural network are:'
#    print '       Input-to-Hidden '
#    print wWeightArray
#    print '       Hidden-to-Output'
#    print vWeightArray
#    print ' '
#    print 'The initial bias weights for this neural network are:'
#    print '        Hidden Bias = ', biasHiddenWeightArray                         
#    print '        Output Bias = ', biasOutputWeightArray
  

          
####################################################################################################
# Before we start training, get a baseline set of outputs, errors, and SSE 
####################################################################################################                
                            
    print ' '
    print '  Before training:'
   
    ComputeOutputsAcrossAllTrainingData (alpha, numTrainingDataSets, wWeightArray0, biasHiddenWeightArray0, 
    vWeightArray0, biasOutputWeightArray0, wWeightArray1, biasHiddenWeightArray1, 
    vWeightArray1, biasOutputWeightArray1, wWeightArray2, biasHiddenWeightArray2, 
    vWeightArray2, biasOutputWeightArray2, wWeightArray3, biasHiddenWeightArray3, 
    vWeightArray3, biasOutputWeightArray3, wWeightArray4, biasHiddenWeightArray4, 
    vWeightArray4, biasOutputWeightArray4, GB1wWeightArray, GB1wBiasWeightArray, GB1vWeightArray, GB1vBiasWeightArray) 
                                                                             
          
####################################################################################################
# Next step - Obtain a single set of randomly-selected training values for alpha-classification 
####################################################################################################                
  
  
    while iteration < maxNumIterations:           

# Increment the iteration count
        iteration = iteration +1

# For any given pass, we re-initialize the training list
        trainingDataList = (
        0,0,0,0,0,0,0,0,0,0,0, 
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0, ' ')                          
                                                                                          
# Randomly select one of four training sets; the inputs will be randomly assigned to 0 or 1
        dataSet = random.randint(1, numTrainingDataSets)
        
        

# We return the list from the function, with values placed inside the list.           

        trainingDataList = obtainSelectedAlphabetTrainingValues (dataSet)  
    

#################################################################################################### 
# STEP 1: Push the input data through Grey Box 1 (GB1)          
####################################################################################################                   
                
        GB1inputDataList = []      
        GB1inputDataArray =  np.zeros(GB1inputArrayLength)
        
        thisTrainingDataList = list()                                                                            
        thisTrainingDataList = trainingDataList[1]
        for node in range(GB1inputArrayLength): 
            
            trainingDataSetNoise=0                     # Initialize random noise variable
            trainingDataSetNoise = np.random.random()  # Assign a random value to the noise variable
            if trainingDataSetNoise < gamma:             # Determine if pixel should be changed using gamma
                
                if thisTrainingDataList[node] == 0:   #  Determine if input pixe is a 0
                    trainingData = 1                   # If input pixel is 0 change to 1
                else: 
                    trainingData = 0                   # If input pixe is 1 change to 0
            else:
                trainingData = thisTrainingDataList[node]    # Keep input pixel if greater than gamma
                        
            GB1inputDataList.append(trainingData)

        GB1desiredOutputArray = np.zeros(GB1outputArrayLength)    # iniitalize the output array with 0's
        GB1desiredClass = trainingDataList[4]                 # identify the desired class
        GB1desiredOutputArray[GB1desiredClass] = 1                # set the desired output for that class to 1 

        GB1hiddenArray = ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray, GB1wWeightArray, GB1wBiasWeightArray)
    
#        print ' '
#        print ' The hidden node activations are:'
#        print hiddenArray

        GB1outputArray = ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray,GB1vWeightArray, GB1vBiasWeightArray)
    
#        print ' '
#        print ' The output node activations are:'
#        print outputArray                                      
                                                                          
                                                                                                                                                    
####################################################################################################
# STEP 2: Create a masked version of the original input
####################################################################################################     
    
# The next step will be to create a padded version of this letter
#    (Expand boundaries by one pixel all around)
        expandedLetterArray = list()
        ExpandTrainingDataList = GB1inputDataList
        expandedLetterArray = expandLetterBoundaries (ExpandTrainingDataList)

# Optional print/debug
#        printExpandedLetter (expandedLetterArray)
    
        mask1LetterArray = maskLetterFunc(expandedLetterArray, mask1)
        mask1LetterList = convertArrayToList(mask1LetterArray)
                 
        mask2LetterArray = maskLetterFunc(expandedLetterArray, mask2)
        mask2LetterList = convertArrayToList(mask2LetterArray)
          

                                                            
####################################################################################################
# Step 3: Create the new input array, combining results from GB1 together with the masking filter result(s)
####################################################################################################                

# In this version, we ARE using the masking filter inputs in conjunction with GB1 results. This means
#   that we need to replace the inputDataArray with masking field data, mask1LetterArray. 
# The important thing now is to first create the new inputDataList that will contain inputs from BOTH 
#   the masked data AND the GB1 outputs. We will use mask1LetterList for this. 

# Also, for this version of the multi-component NN, we're using FOUR masking field inputs, but they're all
#   the same masking field - we're just getting the inputs organized. 
# We previously used the command: arraySizeList = obtainNeuralNetworkSizeSpecs ()
# This returned an arraySizeList that gave us inputArrayLength = 333, 
#    which is 4 x 81 = 324 inputs from four masking fields, plus 9 inputs from the GB1 outputs
#    so that 324 + 9 = 333. 

# We need to add all four masking field inputs into the new inputDataList, and then the GB1 outputs also. 

# First, obtain a full input vector    
# Fill the first part of the training data list with the result of the first masking field

        inputDataList = []      
        inputDataArray =  np.zeros(inputArrayLength) 
        
      
        inputDataList = GB1inputDataList   # Include the original data set with noise

# Add in the results of all 4 masking fields
#   the elements of the input data list. (We'll create new Masking Fields in the next round of code.)
        for node in range(GB1inputArrayLength): # Recall that GB1inputArrayLength is 81
            trainingData = mask1LetterList[node]  # This is the new input from the masking field results
            inputDataList.append(trainingData)
        for node in range(GB1inputArrayLength): # Second set of 81 nodes
            trainingData = mask2LetterList[node]  # This is the second set of input from the masking field results
            inputDataList.append(trainingData)          



# Fill the second part of the training data list with the outputs from GB1          
        for node in range(GB1outputArrayLength): 
            trainingData = GB1outputArray[node]  
            inputDataList.append(trainingData)
         

# Create an input array with both the original training data and the outputs from GB1
        for node in range(inputArrayLength): 
            inputDataArray[node] = inputDataList[node]            
 
          
####################################################################################################
# Step 4: Create the new desired output array, using the full number of classes in the input data
####################################################################################################                

# Note: Earlier, the "desired class" was for the big shape (element 4 in the trainingDataList); 
#       Now, the "desired class" is the final classification into an alphabetic character

        desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
        desiredClass = trainingDataList[2]                 # identify the desired class
        desiredOutputArray[desiredClass] = 1                                       
          
####################################################################################################
# Step 5: Do backpropagation training using the combined (GB1 + MF) inputs based on Class
####################################################################################################                


        
        if trainingDataList[4]==0:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            hiddenArray0 = ComputeSingleFeedforwardPassFirstStep0 (alpha, inputDataArray, wWeightArray0, biasHiddenWeightArray0)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray0

            outputArray = ComputeSingleFeedforwardPassSecondStep0 (alpha, hiddenArray0,vWeightArray0, biasOutputWeightArray0)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray    

        elif trainingDataList[4]==1:
            
            hiddenArray1 = ComputeSingleFeedforwardPassFirstStep1 (alpha, inputDataArray, wWeightArray1, biasHiddenWeightArray1)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray1

            outputArray = ComputeSingleFeedforwardPassSecondStep1 (alpha, hiddenArray1,vWeightArray1, biasOutputWeightArray1)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray            

        elif trainingDataList[4]==2:
            
            hiddenArray2 = ComputeSingleFeedforwardPassFirstStep2 (alpha, inputDataArray, wWeightArray2, biasHiddenWeightArray2)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray2

            outputArray = ComputeSingleFeedforwardPassSecondStep1 (alpha, hiddenArray2,vWeightArray2, biasOutputWeightArray2)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray  

        elif trainingDataList[4]==3:
            
            hiddenArray3 = ComputeSingleFeedforwardPassFirstStep3 (alpha, inputDataArray, wWeightArray3, biasHiddenWeightArray3)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray1

            outputArray = ComputeSingleFeedforwardPassSecondStep3 (alpha, hiddenArray3,vWeightArray3, biasOutputWeightArray3)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray  

        else:
            
            hiddenArray4 = ComputeSingleFeedforwardPassFirstStep4 (alpha, inputDataArray, wWeightArray4, biasHiddenWeightArray4)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray1

            outputArray = ComputeSingleFeedforwardPassSecondStep4 (alpha, hiddenArray4,vWeightArray4, biasOutputWeightArray4)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray     

 
# Initialize the error array
        errorArray = np.zeros(outputArrayLength) 
    
# Determine the error between actual and desired outputs        
        newSSE = 0.0
        for node in range(outputArrayLength):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

#        print ' '
#        print ' The error values are:'
#        print errorArray   
        
# Print the Summed Squared Error  
#        print 'Initial SSE = %.6f' % newSSE
#        SSE = newSSE

         
          
####################################################################################################
# Perform backpropagation
####################################################################################################                
                
        if trainingDataList[4]==0:

# Perform first part of the backpropagation of weight changes    
            newVWeightArray0 = backpropagateOutputToHidden0 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray0, vWeightArray0)
            newBiasOutputWeightArray0 = backpropagateBiasOutputWeights0 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray0) 

# Perform first part of the backpropagation of weight changes       
            newWWeightArray0 = backpropagateHiddenToInput0 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray0,
            inputDataList, vWeightArray0, wWeightArray0, biasHiddenWeightArray0, biasOutputWeightArray0)

            newBiasHiddenWeightArray0 = backpropagateBiasHiddenWeights0 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray0,
            inputDataList, vWeightArray0, wWeightArray0, biasHiddenWeightArray0, biasOutputWeightArray0)  
    
                    
# Assign new values to the weight matrices
# Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
            vWeightArray0 = newVWeightArray0[:]
    
            biasOutputWeightArray0 = newBiasOutputWeightArray0[:]
    
# Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
            wWeightArray0 = newWWeightArray0[:]  
    
            biasHiddenWeightArray0 = newBiasHiddenWeightArray0[:] 
    
# Compute a forward pass, test the new SSE                                                                                
                                                                                                                                    
            hiddenArray0 = ComputeSingleFeedforwardPassFirstStep0 (alpha, inputDataArray, wWeightArray0, biasHiddenWeightArray0)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray

            outputArray = ComputeSingleFeedforwardPassSecondStep0 (alpha, hiddenArray0, vWeightArray0, biasOutputWeightArray0)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray   

        elif trainingDataList[4]==1:
            
# Perform first part of the backpropagation of weight changes    
            newVWeightArray1 = backpropagateOutputToHidden1 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray1, vWeightArray1)
            newBiasOutputWeightArray1 = backpropagateBiasOutputWeights1 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray1) 

# Perform first part of the backpropagation of weight changes       
            newWWeightArray1 = backpropagateHiddenToInput1 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray1,
            inputDataList, vWeightArray1, wWeightArray1, biasHiddenWeightArray1, biasOutputWeightArray1)

            newBiasHiddenWeightArray1 = backpropagateBiasHiddenWeights1 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray1,
            inputDataList, vWeightArray1, wWeightArray1, biasHiddenWeightArray1, biasOutputWeightArray1)  
    
                    
# Assign new values to the weight matrices
# Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
            vWeightArray1 = newVWeightArray1[:]
    
            biasOutputWeightArray1 = newBiasOutputWeightArray1[:]
    
# Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
            wWeightArray1 = newWWeightArray1[:]  
    
            biasHiddenWeightArray1 = newBiasHiddenWeightArray1[:] 
    
# Compute a forward pass, test the new SSE                                                                                
                                                                                                                                    
            hiddenArray1 = ComputeSingleFeedforwardPassFirstStep1 (alpha, inputDataArray, wWeightArray1, biasHiddenWeightArray1)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray

            outputArray = ComputeSingleFeedforwardPassSecondStep1 (alpha, hiddenArray1, vWeightArray1, biasOutputWeightArray1)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray    

        elif trainingDataList[4]==2:
            
# Perform first part of the backpropagation of weight changes    
            newVWeightArray2 = backpropagateOutputToHidden2 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray2, vWeightArray2)
            newBiasOutputWeightArray2 = backpropagateBiasOutputWeights2 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray2) 

# Perform first part of the backpropagation of weight changes       
            newWWeightArray2 = backpropagateHiddenToInput2 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray2,
            inputDataList, vWeightArray2, wWeightArray2, biasHiddenWeightArray2, biasOutputWeightArray2)

            newBiasHiddenWeightArray2 = backpropagateBiasHiddenWeights2 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray2,
            inputDataList, vWeightArray2, wWeightArray2, biasHiddenWeightArray2, biasOutputWeightArray2)  
    
                    
# Assign new values to the weight matrices
# Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
            vWeightArray2 = newVWeightArray2[:]
    
            biasOutputWeightArray2 = newBiasOutputWeightArray2[:]
    
# Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
            wWeightArray2 = newWWeightArray2[:]  
    
            biasHiddenWeightArray2 = newBiasHiddenWeightArray2[:] 
    
# Compute a forward pass, test the new SSE                                                                                
                                                                                                                                    
            hiddenArray2 = ComputeSingleFeedforwardPassFirstStep2 (alpha, inputDataArray, wWeightArray2, biasHiddenWeightArray2)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray

            outputArray = ComputeSingleFeedforwardPassSecondStep2 (alpha, hiddenArray2, vWeightArray2, biasOutputWeightArray2)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray    


        elif trainingDataList[4]==3:
            
# Perform first part of the backpropagation of weight changes    
            newVWeightArray3 = backpropagateOutputToHidden3 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray3, vWeightArray3)
            newBiasOutputWeightArray3 = backpropagateBiasOutputWeights3 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray3) 

# Perform first part of the backpropagation of weight changes       
            newWWeightArray3 = backpropagateHiddenToInput3 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray3,
            inputDataList, vWeightArray3, wWeightArray3, biasHiddenWeightArray3, biasOutputWeightArray3)

            newBiasHiddenWeightArray3 = backpropagateBiasHiddenWeights3 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray3,
            inputDataList, vWeightArray3, wWeightArray3, biasHiddenWeightArray3, biasOutputWeightArray3)  
    
                    
# Assign new values to the weight matrices
# Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
            vWeightArray3 = newVWeightArray3[:]
    
            biasOutputWeightArray3 = newBiasOutputWeightArray3[:]
    
# Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
            wWeightArray3 = newWWeightArray3[:]  
    
            biasHiddenWeightArray3 = newBiasHiddenWeightArray3[:] 
    
# Compute a forward pass, test the new SSE                                                                                
                                                                                                                                    
            hiddenArray3 = ComputeSingleFeedforwardPassFirstStep3 (alpha, inputDataArray, wWeightArray3, biasHiddenWeightArray3)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray

            outputArray = ComputeSingleFeedforwardPassSecondStep3 (alpha, hiddenArray3, vWeightArray3, biasOutputWeightArray3)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray    


        else:
            
# Perform first part of the backpropagation of weight changes    
            newVWeightArray4 = backpropagateOutputToHidden4 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray4, vWeightArray4)
            newBiasOutputWeightArray4 = backpropagateBiasOutputWeights4 (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray4) 

# Perform first part of the backpropagation of weight changes       
            newWWeightArray4 = backpropagateHiddenToInput4 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray4,
            inputDataList, vWeightArray4, wWeightArray4, biasHiddenWeightArray4, biasOutputWeightArray4)

            newBiasHiddenWeightArray4 = backpropagateBiasHiddenWeights4 (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray4,
            inputDataList, vWeightArray4, wWeightArray4, biasHiddenWeightArray4, biasOutputWeightArray4)  
    
                    
# Assign new values to the weight matrices
# Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
            vWeightArray4 = newVWeightArray4[:]
    
            biasOutputWeightArray4 = newBiasOutputWeightArray4[:]
    
# Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
            wWeightArray4 = newWWeightArray4[:]  
    
            biasHiddenWeightArray4 = newBiasHiddenWeightArray4[:] 
    
# Compute a forward pass, test the new SSE                                                                                
                                                                                                                                    
            hiddenArray4 = ComputeSingleFeedforwardPassFirstStep4 (alpha, inputDataArray, wWeightArray4, biasHiddenWeightArray4)
    
#            print ' '
#            print ' The hidden node activations are:'
#            print hiddenArray

            outputArray = ComputeSingleFeedforwardPassSecondStep4 (alpha, hiddenArray4, vWeightArray4, biasOutputWeightArray4)
    
#            print ' '
#            print ' The output node activations are:'
#            print outputArray    


    
# Determine the error between actual and desired outputs

        newSSE = 0.0
        for node in range(outputArrayLength):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

#        print ' '
#        print ' The error values are:'
#        print errorArray   
        
# Print the Summed Squared Error  
#        print 'Previous SSE = %.6f' % SSE
#        print 'New SSE = %.6f' % newSSE 
    
#        print ' '
#        print 'Iteration number ', iteration
#        iteration = iteration + 1

        if newSSE < epsilon:
            print ' '
            print ' *************************************************'
            print ' meeting stopping criterion'
            print ' SSE = ', newSSE, ' for trainingDataList[0]', trainingDataList[2] +1, ' with letter ', trainingDataList[3]

            print ' '
 
            if trainingDataList[4]==0:
                hiddenArray0 = ComputeSingleFeedforwardPassFirstStep (alpha, inputDataArray, wWeightArray0, biasHiddenWeightArray0)

                print ' '
                print ' The hidden node activations are:'
                print hiddenArray0

                outputArray = ComputeSingleFeedforwardPassSecondStep0 (alpha, hiddenArray0, vWeightArray0, biasOutputWeightArray0)
                                
            elif trainingDataList[4]==1:
                hiddenArray1 = ComputeSingleFeedforwardPassFirstStep1 (alpha, inputDataArray, wWeightArray1, biasHiddenWeightArray1)

                print ' '
                print ' The hidden node activations are:'
                print hiddenArray1                

                outputArray = ComputeSingleFeedforwardPassSecondStep1 (alpha, hiddenArray1, vWeightArray1, biasOutputWeightArray1)

    
            elif trainingDataList[4]==2:
                hiddenArray2 = ComputeSingleFeedforwardPassFirstStep2 (alpha, inputDataArray, wWeightArray2, biasHiddenWeightArray2)

                print ' '
                print ' The hidden node activations are:'
                print hiddenArray2                

                outputArray = ComputeSingleFeedforwardPassSecondStep2 (alpha, hiddenArray2, vWeightArray2, biasOutputWeightArray2)        
                
            elif trainingDataList[4]==3:
                hiddenArray3 = ComputeSingleFeedforwardPassFirstStep3 (alpha, inputDataArray, wWeightArray3, biasHiddenWeightArray3)

                print ' '
                print ' The hidden node activations are:'
                print hiddenArray3                

                outputArray = ComputeSingleFeedforwardPassSecondStep3 (alpha, hiddenArray3, vWeightArray3, biasOutputWeightArray3)    
            

            else:
                hiddenArray4 = ComputeSingleFeedforwardPassFirstStep4 (alpha, inputDataArray, wWeightArray4, biasHiddenWeightArray4)

                print ' '
                print ' The hidden node activations are:'
                print hiddenArray4                

                outputArray = ComputeSingleFeedforwardPassSecondStep4 (alpha, hiddenArray4, vWeightArray4, biasOutputWeightArray4)                     
                                                            
            print ' '
            print ' The output node activations are:'
            print outputArray   

            desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
            desiredClass = trainingDataList[2]                 # identify the desired class
            desiredOutputArray[desiredClass] = 1                # set the desired output for that class to 1
     
            print ' '
            print ' The desired output array values are: '
            print desiredOutputArray              
            print ' '
            print ' after running Grey Box 1'
            print ' '
            print '   GB1hiddenArray:' 
            print GB1hiddenArray         
            print ' '
            print '   GB1outputArray:' 
            print GB1outputArray             
            print ' *************************************************'                                    
            break
    print 'Out of while loop at iteration ', iteration 
    
####################################################################################################
# After training, get a new comparative set of outputs, errors, and SSE 
####################################################################################################                           

    print ' '
    print '  After training:'                  
                                                      
    ComputeOutputsAcrossAllTrainingData (alpha, numTrainingDataSets, wWeightArray0, 
    biasHiddenWeightArray0, vWeightArray0, biasOutputWeightArray0,  wWeightArray1, 
    biasHiddenWeightArray1, vWeightArray1, biasOutputWeightArray1, wWeightArray2, 
    biasHiddenWeightArray2, vWeightArray2, biasOutputWeightArray2, wWeightArray3, 
    biasHiddenWeightArray3, vWeightArray3, biasOutputWeightArray3, wWeightArray4, 
    biasHiddenWeightArray4, vWeightArray4, biasOutputWeightArray4, GB1wWeightArray, GB1wBiasWeightArray, 
    GB1vWeightArray, GB1vBiasWeightArray) 

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                                              
####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################                
    
if __name__ == "__main__": main()

####################################################################################################
# End program
#################################################################################################### 

