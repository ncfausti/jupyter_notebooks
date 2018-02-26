
# coding: utf-8

# ## Understand Decision Tree Learning
# 
# 
# Attribute | Val | Play outside = yes | Play outside = no  
# Sunny yes 20 1  
# Sunny no 15 14  
# Snow no 10 5  
# Snow yes 25 10
# 

# ### ID3 / Info. Gain
# Pick the root node, the one we gain the most info from
# The goal is to have the resulting decision tree as small as
# possible (Occam’s Razor)
# 
# 
# S is set of examples  
# P+ is proportion of positive examples in S  
# P_ is proportion of negative examples in S  
# 
# 
# if all examples are P+, entropy is 0  
# if half are P+ and half are P_ entropy is 1  
# 
# <img src="entropy.png" />
# 
# <img src="info_gain.png" />
# #### Where:
# Sv is the subset of S for which attribute a has value v, and  
# the entropy of partitioning the data is calculated by weighing the  
# entropy of each partition by its size relative to the original set
# 
# #### Example from Above
# Attribute | Val | Play outside = yes | Play outside = no  
# Sunny yes 20 1  
# Sunny no 15 14  
# Snow no 10 5  
# Snow yes 25 10
# 
# What is the Entropy?  
# What is the gain for each?  
# ##### max(gain) == root node

# In[4]:


#
# My implementation of entropy
#

import math

# example_set is a list of probabilities
def ent(example_set):
    total = 0
    for prob in example_set:
        try:
            total += prob * math.log(prob,2)
        except ValueError:
            continue
    return -1. * total
print(ent([.5,.5]))

### Example from Udacity Intro. to ML 

print("Entropy p1: ", -35/50*math.log(35/50,2) - 35/50*math.log(35/50,2))


print("entropy: ")
print(-.25*math.log(.25,2) - .75*math.log(.75,2))

1 - (3/4) * .9184


# ### Example from Udacity Intro. to ML 
# <img src="entropy_info_gain.png" />

# In[5]:


#
# Scipy's implementation of entropy
#
# scipy uses log base e, instead of base2, 
# both are fine, just need to pick one and be consistent
#

import scipy.stats as stat
stat.entropy([.7,.3])


# #### Dataset (counts) for Part 1   
# <img src="dataset1.png" />  
# 
# What is the entropy of the dataset?  
# What is the Information Gain for each?  
# What is the root node?  
# * Going to split on Sunny = yes/no OR Snowy = yes/no
# 
# 
# ### The best split should make the samples "pure" ie. homogeneous

# In[3]:


# 50 Instances
TOTAL = 50.

PLAY = 35
NO_PLAY = 15

PLAY_PCT = 1. * (PLAY / TOTAL)
NO_PLAY_PCT = 1. * (NO_PLAY / TOTAL)

ent = stat.entropy([PLAY_PCT, NO_PLAY_PCT])

# You can see that there are 50 instances in the data set, 35 of which
# are positive (Play outside = yes) and the remaining 15 are negative (Play
# outside= no)

print("PlayPct: %f" % PLAY_PCT)
print("NoPlayPct: %f" % NO_PLAY_PCT)
print("Entropy: %.4f" % ent)

# Sunny/not sunny
sunny = {
"sunYesPlayYes" : 20,
"sunYesPlayNo" : 1,
"sunNoPlayYes" : 15,
"sunNoPlayNo" : 14,
}

# Snow/not snow
snowy = {
"snowNoPlayYes" : 10,
"snowNoPlayNo" : 5,
"snowYesPlayYes" : 25,
"snowYesPlayNo" : 10,
}



# ### Root node should be Sunny = yes/no
# ##### provides the largest reduction in entropy

# In[ ]:


# Quetion 1b

# put into recursive formula what I was doing on paper

def induceDTree(data):
    # if all T or all F, set classAttribute
    if len(set(data)) == 1:
        return 
        


# In[6]:


# Part II

from sklearn import tree
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
clf.predict([[2.,2.]])


# In[ ]:


# Part 2.2. Decision Trees as Features
# sklearn, use multiple versions of DTree algorithm
# then use those DTs as features for SGD - Stochastic Gradient Descent

# turnin -c cis519 -p hw1 hw1.pdf test_labels.txt README hw1.py

# BADGES.zip
# 
# badges.modified.data.train --all examples
# 
# badges.modified.data.fold1, fold2, fold3 etc.
# 
# badges.modified.data.test, -task is to label these test_labels.txt
# using best performing model
# submit in same format as training file, each line should start with label
# followed by a space and followed by the actual name

# Main goal, learn function label names with +,- correctly
#   Use DT algorithm to generate features for learning
#   a linear separator.

# INPUT: two lower cased strings - first and last names
# OUTPUT: + or -

# a) FEATURE EXTRACTION AND INSTANCE GENERATION
#     what is INSTANCE GENERATION?
#     need to extract features frome data
#       - extract ten features for each example
#          - boolean feature types

# For example, consider the name “naoki abe” from the data set. Suppose you
# want to extract features corresponding to the first letter “n” in the first name,
# you will have 26 Boolean features, one for each letter in the alphabet. Only
# the one corresponding to n will be 1 and the rest will be 0. This will give us
# 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
# where the 14th element corresponds to the feature String=First,Position=1,Character=n. 
# Note that the features defined earlier are actually feature types. In fact, you will have
# 260 features of the form String=i,Position=j,Character=k, where i can be
# FirstName or LastName, j can be 1, . . . , 5 and k can be a,b,c, ..., z.

# In addition to the feature types described above, feel free to include additional features

# b) Decision Trees and SGD learning algorithms
#     i. use SGD as a baseline  (sklearn.linear model.SGDClassifier)
#         -use five fold cross validation to tune parameters (learning rate, error thresh.)
#          of the SGD algorithm
#            - how do i do five fold CV in sklearn?
#    ii. Grow decision tree: use DT package to train a decision tree with the CART
#         algorithm available in sklearn
#   iii. grow DTs of depth=4, repeat as above limiting depth to 4
#          -Decision trees with limited depth are also called decision stumps.

#    iv. grow DTs of depth= 8, repeat as above limiting depth to 8
#     v. Decision stumps (limited depth DTs) as features, use feature set defined in a)



# 2.3 Evaluation



# In[66]:


def nameToVec(full_name):
    feature_vector = []
    # nick fausti
    # [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
    #  0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    #  0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    #  0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    #  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    #  0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    #  1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    #  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    #  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    #  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # 26 * 10 = 260
    # if either name doesn't have at least 5 chars, pad with 0 until len > 129
    name_list = full_name.split(" ")
    first = name_list[0]
#     print(first)
    last = name_list[1]
#     print(last)
    
    place = 0
    for letter in first:
        if place < 5:
            for i in range(26):
                feature_vector.append(0)
            position = ord(letter) - 97 + (place * 26)
#             print(letter + ":" + str(position))
            feature_vector[position] = 1
            place += 1
    
    # pad 0s for first names less than 5 chars
    while len(feature_vector) < 130:
        feature_vector.append(0)
    
    place = 5
    for letter in last:
        if place < 10:
            for i in range(26):
                feature_vector.append(0)
            position = ord(letter) - 97 + (place * 26)
#             print(letter + ":" + str(position))
            feature_vector[position] = 1
            place += 1
    
    while len(feature_vector) < 260:
        feature_vector.append(0)
    return feature_vector


# In[67]:


# test nameToVec
print(nameToVec("nick fausti"))
nameToVec("nick fausti") == [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]


# In[69]:


# Feature Extraction
from sklearn import tree
# X should be a list of n=260 vectors, 26 * 10 characters (5 in first, 5 in lastname)
# X = [[0, 0], [1, 1]]
names = []
labels = []

X = []

with open("hw1/badges/badges.modified.data.train", "r") as f:
    for line in f:
        data = line.split(" ")
        names.append(data[1] + " " + data[2].strip())
        if data[0] == "+":
            labels.append(1)
        else:
            labels.append(0)

for name in names:
    X.append(nameToVec(name))


print("names: " + str(len(names)))
print("labels: " + str(len(labels)))
print("X: " + str(len(X)))


# y is a boolean label -/+ ie. 0/1
# y = [0, 1]

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X,y)
# clf.predict([[2.,2.]])


# In[80]:


from sklearn.linear_model import SGDClassifier


# In[163]:


import numpy as np
clf = SGDClassifier(loss="log")
clf.fit(np.array(X),labels)


# In[123]:


print(clf.predict([nameToVec("shai bendavid")]))
print(clf.predict([nameToVec("nick fausti")]))
print(clf.predict([nameToVec("david mathias")]))


# In[214]:


# For each algorithm you experiment with, (i) run five-fold cross validation on the given
# data set.
#
# pA is the average accuracy over the five folds
#
# In addition
# to pA, record also the performance of your algorithm on the training data, trA. This
# will also be the average you get over the five training folds. (ii) Calculate the 99%
# confidence interval of this estimate using Student’s t-test.

# SGD

def makeFoldsFromFile(foldNum):
    X = []
    names = []
    labels = []
    
    with open("hw1/badges/badges.modified.data.fold" + str(foldNum), "r") as f:
        for line in f:
            data = line.split(" ")
            names.append(data[1] + " " + data[2].strip())
            if data[0] == "+":
                labels.append(1)
            else:
                labels.append(0)
    
    for name in names:
        X.append(nameToVec(name))
    return (X, names, labels)

# vectors, strings_of_names, labels
X1, test1, y1 = makeFoldsFromFile(1)
X2, test2, y2 = makeFoldsFromFile(2)
X3, test3, y3 = makeFoldsFromFile(3)
X4, test4, y4 = makeFoldsFromFile(4)
X5, test5, y5 = makeFoldsFromFile(5)

# FOLD GROUP 1
fold1Features = X1 + X2 + X3 + X4
fold1Labels = y1 + y2 + y3 + y4
fold1test = test5

# FOLD GROUP 2
fold2Features = X2 + X3 + X4 + X5
fold2Labels = y2 + y3 + y4 + y5
fold2test = test1

# FOLD GROUP 3
fold3Features = X3 + X4 + X5 + X1
fold3Labels = y3 + y4 + y5 + y1
fold3test = test2

# FOLD GROUP 4
fold4Features = X4 + X5 + X1 + X2
fold4Labels = y4 + y5 + y1 + y2
fold4test = test3

# FOLD GROUP 5
fold5Features = X5 + X1 + X2 + X3
fold5Labels = y5 + y1 +y2 + y3
fold5test = test4


# In[216]:


clf1 = clf.fit(fold1Features, fold1Labels)
score1 = clf.score(X5,y5)

clf2 = clf.fit(fold2Features, fold2Labels)
score2 = clf2.score(X1,y1) 

clf3 = clf.fit(fold3Features, fold3Labels)
score3 = clf3.score(X2,y2)

clf4 = clf.fit(fold4Features, fold4Labels)
score4 = clf4.score(X3,y3)

clf5 = clf.fit(fold5Features, fold5Labels)
score5 = clf5.score(X4,y4) 

print(np.average([score1,score2,score3,score4,score5]))


# In[213]:


# print( sklearn.metrics.accuracy_score(predictions, y5) )


# In[169]:


# repeat what I did above, swapping out the 5th test labels for
# others
# average that score, repeat with decision trees, stumps, and using
# stumps as features for SGD


# In[217]:


fold_features_list = [fold1Features,fold2Features,
                 fold3Features,fold4Features,fold5Features]

fold_labels_list = [fold1Labels,fold2Labels,
                 fold3Labels,fold4Labels,fold5Labels]

fold_X = [X5, X1, X2, X3, X4]
fold_y = [y5, y1, y2, y3, y4]

dtMaxScores = []

dtMaxClf = sklearn.tree.DecisionTreeClassifier()

for i, feat in enumerate(fold_features_list):
    dtMaxClf.fit(feat, fold_labels_list[i])
    dtMaxScores.append(dtMaxClf.score(fold_X[i], fold_y[i]))
    


# In[219]:


# No-max decision tree
print(np.average(dtMaxScores))


# In[238]:


dt4Scores = []

dt4clf = sklearn.tree.DecisionTreeClassifier(max_depth=4)

for i, feat in enumerate(fold_features_list):
    dt4clf.fit(feat, fold_labels_list[i])
    dt4Scores.append(dt4clf.score(fold_X[i], fold_y[i]))

print(np.average(dt4Scores))


# In[241]:


# Decision tree with max-depth=8
dt8Scores = []

dt8clf = sklearn.tree.DecisionTreeClassifier(max_depth=8)

for i, feat in enumerate(fold_features_list):
    dt8clf.fit(feat, fold_labels_list[i])
    dt8Scores.append(dt8clf.score(fold_X[i], fold_y[i]))

print(np.average(dt8Scores))


# In[258]:


# Decision Stumps as features
boostNamesVecsLabels = [] #[(name, feature_vec, label)]
boostLabels = [] 

boostX = []

with open("hw1/badges/badges.modified.data.train", "r") as f:
    for line in f:
        name = ""
        label = 0
        
        data = line.split(" ")
        name = data[1] + " " + data[2].strip()
        
        feature_vec = nameToVec(name)
        
        if data[0] == "+":
            label = 1
        else:
            label = 0
        results = [name, feature_vec, label]
        boostNamesVecsLabels.append(results)

# for name in boostNames:
#     boostX.append(nameToVec(name))

# should be a vector of len 100 (100 predictions from
# decisionStubs trained on 50% of data)

# list of DT objects
classifiers = []
wordPredictions = []

for i in range(100):
    # take 700/2 names to use as 50% train sample
    labelDict = {}
    trainNames = []
    trainVecs = []
    
    for i,v in enumerate(boostNamesVecsLabels):
        # set name : label
        labelDict[v[0]] = v[2]
        trainNames.append(v[0])
    
    trainSubset = np.random.choice(trainNames, 350, replace=False)
    
    trainX = []
    trainY = []
    
    for name in trainSubset:
        trainX.append(nameToVec(name))
        trainY.append(labelDict[name])

    clf = sklearn.tree.DecisionTreeClassifier(max_depth=8)
    clf.fit(trainX, trainY)
    
    classifiers.append(clf)


# In[277]:


stumps = []
stumpVec = []

# for name in data.fold1 + data.fold2 + data.fold3 + data.fold4 + data.fold5
for clf in classifiers:
#     stumpVector.append(clf.predict([nameToVec("nick fausti")])[0])
    stumpVec.append(clf.predict([nameToVec("nick fausti")])[0])

stumps.append(stumpVec)

# Now do 5-fold cross validation with new model


# In[282]:


# fold_features_list = [fold1Features,fold2Features,
#                  fold3Features,fold4Features,fold5Features]

# fold_labels_list = [fold1Labels,fold2Labels,
#                  fold3Labels,fold4Labels,fold5Labels]

# fold_X = [X5, X1, X2, X3, X4]
# fold_y = [y5, y1, y2, y3, y4]

# stumpScores = []

# dtMaxClf = sklearn.tree.DecisionTreeClassifier()

# for i, feat in enumerate(fold_features_list):
#     dtMaxClf.fit(feat, fold_labels_list[i])
#     dtMaxScores.append(dtMaxClf.score(fold_X[i], fold_y[i]))
len(fold1Labels)


# In[292]:


def makeStumpFoldsFromFile(foldNum):
    X = []
    names = []
    labels = []
    stumps = []
    
    with open("hw1/badges/badges.modified.data.fold" + str(foldNum), "r") as f:
        for line in f:
            data = line.split(" ")
            names.append(data[1] + " " + data[2].strip())
            if data[0] == "+":
                labels.append(1)
            else:
                labels.append(0)
    
    for name in names:
        stumpVec = []
        X.append(nameToVec(name))
        # append prediction from 100 different classifiers to stumpVec
        for clf in classifiers:
        #     stumpVector.append(clf.predict([nameToVec("nick fausti")])[0])
            stumpVec.append(clf.predict([nameToVec(name)])[0])
        stumps.append(stumpVec)

    return (stumps, X, names, labels)


stumpVec = []

# for name in data.fold1 + data.fold2 + data.fold3 + data.fold4 + data.fold5


stumps.append(stumpVec)

# vectors, strings_of_names, labels
stump1, X1, test1, y1 = makeStumpFoldsFromFile(1)
stump2, X2, test2, y2 = makeStumpFoldsFromFile(2)
stump3, X3, test3, y3 = makeStumpFoldsFromFile(3)
stump4, X4, test4, y4 = makeStumpFoldsFromFile(4)
stump5, X5, test5, y5 = makeStumpFoldsFromFile(5)

# FOLD GROUP 1
fold1StumpFeatures = stump1 + stump2 + stump3 + stump4
fold1Features = X1 + X2 + X3 + X4
fold1Labels = y1 + y2 + y3 + y4
fold1test = test5
fold1StumpTest = stump5

# # FOLD GROUP 2
fold2StumpFeatures = stump2 + stump3 + stump4 + stump5
fold2Features = X2 + X3 + X4 + X5
fold2Labels = y2 + y3 + y4 + y5
fold2test = test1
fold2StumpTest = stump1

# FOLD GROUP 3
fold3StumpFeatures = stump3 + stump4 + stump5 + stump1
fold3Features = X3 + X4 + X5 + X1
fold3Labels = y3 + y4 + y5 + y1
fold3test = test2
fold3StumpTest = stump2

# FOLD GROUP 4
fold4StumpFeatures = stump4 + stump5 + stump1 + stump2
fold4Features = X4 + X5 + X1 + X2
fold4Labels = y4 + y5 + y1 + y2
fold4test = test3
fold4StumpTest = stump3

# FOLD GROUP 5
fold5StumpFeatures = stump5 + stump1 + stump2 + stump3
fold5Features = X5 + X1 + X2 + X3
fold5Labels = y5 + y1 +y2 + y3
fold5test = test4
fold5StumpTest = stump4


# In[286]:


testcase = []
for clf in classifiers:
    testcase.append(clf.predict([nameToVec("neta rivkin")])[0])
stump1[0] == testcase


# In[287]:


clf = SGDClassifier(loss="log")
clf.fit(fold1StumpFeatures, fold1Labels)


# In[288]:


clf.score(stump5, y5)


# In[308]:


stump_cv_features_list = [fold1StumpFeatures,fold2StumpFeatures,
                 fold3StumpFeatures,fold4StumpFeatures,fold5StumpFeatures]

stump_cv_labels_list = [fold1Labels,fold2Labels,
                  fold3Labels,fold4Labels,fold5Labels]

stump_fold_X = [stump5, stump1, stump2, stump3, stump4]
stump_fold_y = [y5, y1, y2, y3, y4]

stumpScores = []

stumpClf = SGDClassifier(loss="log")

for i, feat in enumerate(stump_cv_features_list):
    stumpClf.fit(feat, stump_cv_labels_list[i])
    stumpScores.append(stumpClf.score(stump_fold_X[i], stump_fold_y[i]))


# In[309]:


np.average(stumpScores)


# In[ ]:




