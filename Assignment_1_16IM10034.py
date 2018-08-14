
# coding: utf-8

# In[1]:


import sys
orig_stdout = sys.stdout
f = open('output.txt', 'w')
sys.stdout = f


# In[2]:


import nltk


# In[3]:


from nltk.corpus import brown


# In[4]:


nltk.download('brown')


# In[5]:


sentences = list(brown.sents()[0:40000])


# In[6]:


# Data Pre-Processing
for i in range(len(sentences)):
    sentences[i] = [token.lower() for token in sentences[i]]
for i in range(len(sentences)):
    p=[]
    for j in range(len(sentences[i])):
        if sentences[i][j].isalpha()==True:
            p.append(sentences[i][j])
    sentences[i]=p
for i in range(len(sentences)):
    sentences[i]=' '.join(sentences[i])


# In[7]:


from nltk import bigrams, ngrams, trigrams 


# In[8]:


# Verification of Zipf's Law for Unigrams
unigrams=[]
for elem in sentences:
    unigrams.extend(elem.split())
from nltk.probability import FreqDist
fdist = FreqDist(unigrams)
import pylab
import math
from scipy import stats
words = fdist.most_common()
x = [math.log10(i[1]) for i in words]
y = [math.log10(i) for i in range(1, len(x))]
x.pop()
(m, b) = pylab.polyfit(x, y, 1)
yp = pylab.polyval([m, b], x)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print('Slope and Intercept for log-log plot for zipfs law in unigrams')
print(slope,intercept)


# In[9]:


# Plot for Unigram Zipf's Law
pylab.plot(x, yp)
pylab.scatter(x, y)
pylab.ylim([min(y), max(y)])
pylab.xlim([min(x), max(x)])
pylab.grid(True)
pylab.ylabel('Counts of words')
pylab.xlabel('Ranks of words')
pylab.show()


# In[10]:


# Top 10 Unigrams
print('Top 10 Unigrams with count')
print(words[0:10])


# In[11]:


# Function to Make Bigram Model from Sentences
def bigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent.split(),2, pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    return model


# In[12]:


# Verification of Zipf's Law for Bigrams
bigram=bigram_model(sentences)
built_bigram={}
for i in bigram.keys():
    for j in bigram[i].keys():
        built_bigram[str(i)+' '+str(j)]=bigram[i][j]
from nltk.probability import FreqDist
fdist2 = FreqDist(built_bigram)
import pylab
import math
from scipy import stats
words = fdist2.most_common()
x = [math.log10(i[1]) for i in words]
y = [math.log10(i) for i in range(1, len(x))]
x.pop()
(m, b) = pylab.polyfit(x, y, 1)
yp = pylab.polyval([m, b], x)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print('Slope and Intercept for log-log plot for zipfs law in bigrams')
print(slope,intercept)


# In[13]:


# Plot for Bigram Zipf's Law
pylab.plot(x, yp)
pylab.scatter(x, y)
pylab.ylim([min(y), max(y)])
pylab.xlim([min(x), max(x)])
pylab.grid(True)
pylab.ylabel('Counts of bigrams')
pylab.xlabel('Ranks of bigrams')
pylab.show()


# In[14]:


# None corresponds to the pad
# Top 10 Bigrams with None allowed
print('Top 10 Bigrams with count when None allowed')
print(words[0:10])


# In[15]:


# Top 10 Bigrams without None 
print('Top 10 Bigrams with count when None not allowed')
ans=[]
for i in words:
    if 'None' not in i[0]:
        ans.append(i)
    if len(ans)==10:
        break
print(ans)


# In[16]:


# Function to make Trigram Model from Sentences
def trigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent.split(),3, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1
    return model


# In[17]:


# Verification of Zipf's Law for Trigrams
trigram=trigram_model(sentences)
built_trigram={}
for i in trigram.keys():
    for j in trigram[i].keys():
        built_trigram[str(i[0])+' '+str(i[1])+' '+str(j)]=trigram[i][j]
from nltk.probability import FreqDist
fdist3 = FreqDist(built_trigram)
import pylab
import math
from scipy import stats
words = fdist3.most_common()
x = [math.log10(i[1]) for i in words]
y = [math.log10(i) for i in range(1, len(x))]
x.pop()
(m, b) = pylab.polyfit(x, y, 1)
yp = pylab.polyval([m, b], x)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print('Slope and Intercept for log-log plot for zipfs law in trigrams')
print(slope,intercept)


# In[18]:


# Zipf's Law plot for Trigrams
pylab.plot(x, yp)
pylab.scatter(x, y)
pylab.ylim([min(y), max(y)])
pylab.xlim([min(x), max(x)])
pylab.grid(True)
pylab.ylabel('Counts of trigrams')
pylab.xlabel('Ranks of trigrams')
pylab.show()


# In[19]:


# None corresponds to the pad
# Top 10 Trigrams with None allowed
print('Top 10 Trigrams with count when None allowed')
print(words[0:10])


# In[20]:


# Top 10 Trigrams without None 
print('Top 10 Trigrams with count when None not allowed')
ans=[]
for i in words:
    if 'None' not in i[0]:
        ans.append(i)
    if len(ans)==10:
        break
print(ans)


# In[21]:


# Converting Models of Count to Models of Probabilities

from collections import Counter
unigram=Counter(unigrams)
unigram_total=len(unigrams)

for word in unigram:
    unigram[word]/=unigram_total

for w1 in bigram:
        tot_count=float(sum(bigram[w1].values()))
        for w2 in bigram[w1]:
            bigram[w1][w2]/=tot_count

for (w1,w2) in trigram:
        tot_count=float(sum(trigram[(w1,w2)].values()))
        for w3 in trigram[(w1,w2)]:
            trigram[(w1,w2)][w3]/=tot_count


# In[22]:


text_file = open("test_examples.txt", "r")
test_sentences=text_file.read().split('\n')
# test_sentences=['he lived a good life','the man was happy','the person was good','the girl was sad','he won the war']


# In[23]:


print("Unigram Test")
#computes perplexity of the unigram model on a testset  
def uniperplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(uniperplexity(i, unigram)))


# In[24]:


print('Bigram Test')
#computes perplexity of the bigram model on a testset  
def biperplexity(testset, model):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            model[testset[i-1]][testset[i]]
        except KeyError:
            perplexity=math.inf
            break
        perplexity = perplexity * (1/model[testset[i-1]][testset[i]])
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(biperplexity(i, bigram)))


# In[25]:


print('Trigram Test')
#computes perplexity of the trigram model on a testset  
def triperplexity(testset, model):
    testset = testset.split()
    testset = [None] + [None] + testset + [None] + [None] 
    perplexity = 1
    N = 0
    for i in range(2,len(testset)):
        N += 1
        try:
            model[(testset[i-2],testset[i-1])][testset[i]]
        except KeyError:
            perplexity=math.inf
            break
        perplexity = perplexity * (1/model[(testset[i-2],testset[i-1])][testset[i]])
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(triperplexity(i, trigram)))


# In[26]:


# Calculating Log likehoods of each model on Test Data
import numpy as np

test_unigram_arr=[]
print('Unigram test \n')
for elem in test_sentences:
    p_val=np.sum([math.log(unigram[i]) for i in elem.split()])
    test_unigram_arr.append(p_val)
    print('The sequence '+elem+' has unigram log-likelihood of '+ str(round(p_val,4)))


print('\nBigram test \n')
test_bigram_arr=[]
for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            bigram[w1][w2]
        except KeyError:
            p_val=-1*math.inf
            break
        p_val+=math.log(bigram[w1][w2])
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,4)))
    
    test_bigram_arr.append(p_val)


test_trigram_arr=[]
print('\nTrigram test \n')
for elem in test_sentences:
    p_val=0
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(trigram[(w1,w2)][w3])
        except Exception as e:
            p_val=-1*math.inf
            break
    print('The sequence '+ elem +' has trigram log-likelihood of '+ str(round(p_val,4)))
    
    test_trigram_arr.append(p_val)
            


# In[27]:


# Counts Models
bigram_counts=bigram_model(sentences)
trigram_counts=trigram_model(sentences)


# In[28]:


# Additive Smoothing Models for k=1
bi_ls=bigram_model(sentences)
tri_ls=trigram_model(sentences)
k=1
from collections import Counter
uni_ls=Counter(unigrams)
unigram_total=len(unigrams)
u=len(set(unigrams))
for word in uni_ls:
    uni_ls[word]=(k+uni_ls[word])/(unigram_total+k*u)
ubp=(k)/(unigram_total+k*u)

for w1 in bi_ls:
        tot_count=float(sum(bi_ls[w1].values()))
        for w2 in bi_ls[w1]:
            bi_ls[w1][w2]=(k+bi_ls[w1][w2])/(tot_count+k*u)

for (w1,w2) in tri_ls:
        tot_count=float(sum(tri_ls[(w1,w2)].values()))
        for w3 in tri_ls[(w1,w2)]:
            tri_ls[(w1,w2)][w3]=(k+tri_ls[(w1,w2)][w3])/(k*u+tot_count)

print('Additive Smoothing with k='+str(k))

import numpy as np

test_unigram_arr=[]

print('Unigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=np.sum([math.log(uni_ls[i]) for i in elem.split()])
    test_unigram_arr.append(p_val)
    print('The sequence '+elem+' has unigram log-likelihood of '+ str(round(p_val,8)))


print('\nBigram test with Additive Smoothing\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(bi_ls[w1][w2])
        except Exception as e:
            p_val+=math.log(k/(float(sum(bigram_counts[w1].values()))+k*u))
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,8)))
    
    test_bigram_arr.append(p_val)


test_trigram_arr=[]
print('\nTrigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=0
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(tri_ls[(w1,w2)][w3])
        except Exception as e:
            try:
                p_val+=math.log((k/(k*u+float(sum(trigram_counts[(w1,w2)].values())))))
            except Exception as e:
                p_val+=math.log((k/(k*u)))
    print('The sequence '+ elem +' has trigram log-likelihood of '+ str(round(p_val,8)))
    
    test_trigram_arr.append(p_val)
     
print('\nUnigram Perplexity with Additive Smoothing')
for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(uniperplexity(i, uni_ls)))
    
print('\nBigram Perplexity with Additive Smoothing')
#computes perplexity of the bigram model on a testset  
def mbiperplexity(testset, model):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            perplexity = perplexity * (1/model[testset[i-1]][testset[i]])
        except KeyError:
            perplexity = perplexity * (1/(k/(float(sum(bigram_counts[w1].values()))+k*u)))
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(mbiperplexity(i, bi_ls)))

print('\nTrigram Perplexity with Additive Smoothing')
#computes perplexity of the trigram model on a testset  
def ttriperplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for i in range(2,len(testset)):
        N += 1
        try:
            perplexity = perplexity * pow((1/model[(testset[i-2],testset[i-1])][testset[i]]),1/float(N))
        except KeyError:
            try:
                perplexity = perplexity * pow((1/(k/(k*u+float(sum(trigram_counts[(w1,w2)].values()))))),1/float(N))
            except Exception as e:
                perplexity = perplexity * pow((1/(k/(k*u))),1/float(N))
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(ttriperplexity(i, tri_ls)))


# In[29]:


# Additive Smoothing Models for k=0.1
bi_ls=bigram_model(sentences)
tri_ls=trigram_model(sentences)
k=0.1
from collections import Counter
uni_ls=Counter(unigrams)
unigram_total=len(unigrams)
u=len(set(unigrams))
for word in uni_ls:
    uni_ls[word]=(k+uni_ls[word])/(unigram_total+k*u)
ubp=(k)/(unigram_total+k*u)

for w1 in bi_ls:
        tot_count=float(sum(bi_ls[w1].values()))
        for w2 in bi_ls[w1]:
            bi_ls[w1][w2]=(k+bi_ls[w1][w2])/(tot_count+k*u)

for (w1,w2) in tri_ls:
        tot_count=float(sum(tri_ls[(w1,w2)].values()))
        for w3 in tri_ls[(w1,w2)]:
            tri_ls[(w1,w2)][w3]=(k+tri_ls[(w1,w2)][w3])/(k*u+tot_count)

print('Additive Smoothing with k='+str(k))

import numpy as np

test_unigram_arr=[]

print('Unigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=np.sum([math.log(uni_ls[i]) for i in elem.split()])
    test_unigram_arr.append(p_val)
    print('The sequence '+elem+' has unigram log-likelihood of '+ str(round(p_val,8)))


print('\nBigram test with Additive Smoothing\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(bi_ls[w1][w2])
        except Exception as e:
            p_val+=math.log(k/(float(sum(bigram_counts[w1].values()))+k*u))
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,8)))
    
    test_bigram_arr.append(p_val)


test_trigram_arr=[]
print('\nTrigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=0
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(tri_ls[(w1,w2)][w3])
        except Exception as e:
            try:
                p_val+=math.log((k/(k*u+float(sum(trigram_counts[(w1,w2)].values())))))
            except Exception as e:
                p_val+=math.log((k/(k*u)))
    print('The sequence '+ elem +' has trigram log-likelihood of '+ str(round(p_val,8)))
    
    test_trigram_arr.append(p_val)
     
print('\nUnigram Perplexity with Additive Smoothing')
for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(uniperplexity(i, uni_ls)))
    
print('\nBigram Perplexity with Additive Smoothing')
#computes perplexity of the bigram model on a testset  
def mbiperplexity(testset, model):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            perplexity = perplexity * (1/model[testset[i-1]][testset[i]])
        except KeyError:
            perplexity = perplexity * (1/(k/(float(sum(bigram_counts[w1].values()))+k*u)))
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(mbiperplexity(i, bi_ls)))

print('\nTrigram Perplexity with Additive Smoothing')
#computes perplexity of the trigram model on a testset  
def ttriperplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for i in range(2,len(testset)):
        N += 1
        try:
            perplexity = perplexity * pow((1/model[(testset[i-2],testset[i-1])][testset[i]]),1/float(N))
        except KeyError:
            try:
                perplexity = perplexity * pow((1/(k/(k*u+float(sum(trigram_counts[(w1,w2)].values()))))),1/float(N))
            except Exception as e:
                perplexity = perplexity * pow((1/(k/(k*u))),1/float(N))
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(ttriperplexity(i, tri_ls)))


# In[30]:


# Additive Smoothing Models for k=0.01
bi_ls=bigram_model(sentences)
tri_ls=trigram_model(sentences)
k=0.01
from collections import Counter
uni_ls=Counter(unigrams)
unigram_total=len(unigrams)
u=len(set(unigrams))
for word in uni_ls:
    uni_ls[word]=(k+uni_ls[word])/(unigram_total+k*u)
ubp=(k)/(unigram_total+k*u)

for w1 in bi_ls:
        tot_count=float(sum(bi_ls[w1].values()))
        for w2 in bi_ls[w1]:
            bi_ls[w1][w2]=(k+bi_ls[w1][w2])/(tot_count+k*u)

for (w1,w2) in tri_ls:
        tot_count=float(sum(tri_ls[(w1,w2)].values()))
        for w3 in tri_ls[(w1,w2)]:
            tri_ls[(w1,w2)][w3]=(k+tri_ls[(w1,w2)][w3])/(k*u+tot_count)

print('Additive Smoothing with k='+str(k))

import numpy as np

test_unigram_arr=[]

print('Unigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=np.sum([math.log(uni_ls[i]) for i in elem.split()])
    test_unigram_arr.append(p_val)
    print('The sequence '+elem+' has unigram log-likelihood of '+ str(round(p_val,8)))


print('\nBigram test with Additive Smoothing\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(bi_ls[w1][w2])
        except Exception as e:
            p_val+=math.log(k/(float(sum(bigram_counts[w1].values()))+k*u))
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,8)))
    
    test_bigram_arr.append(p_val)


test_trigram_arr=[]
print('\nTrigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=0
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(tri_ls[(w1,w2)][w3])
        except Exception as e:
            try:
                p_val+=math.log((k/(k*u+float(sum(trigram_counts[(w1,w2)].values())))))
            except Exception as e:
                p_val+=math.log((k/(k*u)))
    print('The sequence '+ elem +' has trigram log-likelihood of '+ str(round(p_val,8)))
    
    test_trigram_arr.append(p_val)
     
print('\nUnigram Perplexity with Additive Smoothing')
for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(uniperplexity(i, uni_ls)))
    
print('\nBigram Perplexity with Additive Smoothing')
#computes perplexity of the bigram model on a testset  
def mbiperplexity(testset, model):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            perplexity = perplexity * (1/model[testset[i-1]][testset[i]])
        except KeyError:
            perplexity = perplexity * (1/(k/(float(sum(bigram_counts[w1].values()))+k*u)))
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(mbiperplexity(i, bi_ls)))

print('\nTrigram Perplexity with Additive Smoothing')
#computes perplexity of the trigram model on a testset  
def ttriperplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for i in range(2,len(testset)):
        N += 1
        try:
            perplexity = perplexity * pow((1/model[(testset[i-2],testset[i-1])][testset[i]]),1/float(N))
        except KeyError:
            try:
                perplexity = perplexity * pow((1/(k/(k*u+float(sum(trigram_counts[(w1,w2)].values()))))),1/float(N))
            except Exception as e:
                perplexity = perplexity * pow((1/(k/(k*u))),1/float(N))
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(ttriperplexity(i, tri_ls)))


# In[31]:


# Additive Smoothing Models for k=0.001
bi_ls=bigram_model(sentences)
tri_ls=trigram_model(sentences)
k=0.001
from collections import Counter
uni_ls=Counter(unigrams)
unigram_total=len(unigrams)
u=len(set(unigrams))
for word in uni_ls:
    uni_ls[word]=(k+uni_ls[word])/(unigram_total+k*u)
ubp=(k)/(unigram_total+k*u)

for w1 in bi_ls:
        tot_count=float(sum(bi_ls[w1].values()))
        for w2 in bi_ls[w1]:
            bi_ls[w1][w2]=(k+bi_ls[w1][w2])/(tot_count+k*u)

for (w1,w2) in tri_ls:
        tot_count=float(sum(tri_ls[(w1,w2)].values()))
        for w3 in tri_ls[(w1,w2)]:
            tri_ls[(w1,w2)][w3]=(k+tri_ls[(w1,w2)][w3])/(k*u+tot_count)

print('Additive Smoothing with k='+str(k))

import numpy as np

test_unigram_arr=[]

print('Unigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=np.sum([math.log(uni_ls[i]) for i in elem.split()])
    test_unigram_arr.append(p_val)
    print('The sequence '+elem+' has unigram log-likelihood of '+ str(round(p_val,8)))


print('\nBigram test with Additive Smoothing\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(bi_ls[w1][w2])
        except Exception as e:
            p_val+=math.log(k/(float(sum(bigram_counts[w1].values()))+k*u))
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,8)))
    
    test_bigram_arr.append(p_val)


test_trigram_arr=[]
print('\nTrigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=0
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(tri_ls[(w1,w2)][w3])
        except Exception as e:
            try:
                p_val+=math.log((k/(k*u+float(sum(trigram_counts[(w1,w2)].values())))))
            except Exception as e:
                p_val+=math.log((k/(k*u)))
    print('The sequence '+ elem +' has trigram log-likelihood of '+ str(round(p_val,8)))
    
    test_trigram_arr.append(p_val)
     
print('\nUnigram Perplexity with Additive Smoothing')
for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(uniperplexity(i, uni_ls)))
    
print('\nBigram Perplexity with Additive Smoothing')
#computes perplexity of the bigram model on a testset  
def mbiperplexity(testset, model):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            perplexity = perplexity * (1/model[testset[i-1]][testset[i]])
        except KeyError:
            perplexity = perplexity * (1/(k/(float(sum(bigram_counts[w1].values()))+k*u)))
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(mbiperplexity(i, bi_ls)))

print('\nTrigram Perplexity with Additive Smoothing')
#computes perplexity of the trigram model on a testset  
def ttriperplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for i in range(2,len(testset)):
        N += 1
        try:
            perplexity = perplexity * pow((1/model[(testset[i-2],testset[i-1])][testset[i]]),1/float(N))
        except KeyError:
            try:
                perplexity = perplexity * pow((1/(k/(k*u+float(sum(trigram_counts[(w1,w2)].values()))))),1/float(N))
            except Exception as e:
                perplexity = perplexity * pow((1/(k/(k*u))),1/float(N))
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(ttriperplexity(i, tri_ls)))


# In[32]:


# Additive Smoothing Models for k=0.0001
bi_ls=bigram_model(sentences)
tri_ls=trigram_model(sentences)
k=0.0001
from collections import Counter
uni_ls=Counter(unigrams)
unigram_total=len(unigrams)
u=len(set(unigrams))
for word in uni_ls:
    uni_ls[word]=(k+uni_ls[word])/(unigram_total+k*u)
ubp=(k)/(unigram_total+k*u)

for w1 in bi_ls:
        tot_count=float(sum(bi_ls[w1].values()))
        for w2 in bi_ls[w1]:
            bi_ls[w1][w2]=(k+bi_ls[w1][w2])/(tot_count+k*u)

for (w1,w2) in tri_ls:
        tot_count=float(sum(tri_ls[(w1,w2)].values()))
        for w3 in tri_ls[(w1,w2)]:
            tri_ls[(w1,w2)][w3]=(k+tri_ls[(w1,w2)][w3])/(k*u+tot_count)

print('Additive Smoothing with k='+str(k))

import numpy as np

test_unigram_arr=[]

print('Unigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=np.sum([math.log(uni_ls[i]) for i in elem.split()])
    test_unigram_arr.append(p_val)
    print('The sequence '+elem+' has unigram log-likelihood of '+ str(round(p_val,8)))


print('\nBigram test with Additive Smoothing\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(bi_ls[w1][w2])
        except Exception as e:
            p_val+=math.log(k/(float(sum(bigram_counts[w1].values()))+k*u))
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,8)))
    
    test_bigram_arr.append(p_val)


test_trigram_arr=[]
print('\nTrigram test with Additive Smoothing\n')
for elem in test_sentences:
    p_val=0
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(tri_ls[(w1,w2)][w3])
        except Exception as e:
            try:
                p_val+=math.log((k/(k*u+float(sum(trigram_counts[(w1,w2)].values())))))
            except Exception as e:
                p_val+=math.log((k/(k*u)))
    print('The sequence '+ elem +' has trigram log-likelihood of '+ str(round(p_val,8)))
    
    test_trigram_arr.append(p_val)
     
print('\nUnigram Perplexity with Additive Smoothing')
for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(uniperplexity(i, uni_ls)))
    
print('\nBigram Perplexity with Additive Smoothing')
#computes perplexity of the bigram model on a testset  
def mbiperplexity(testset, model):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            perplexity = perplexity * (1/model[testset[i-1]][testset[i]])
        except KeyError:
            perplexity = perplexity * (1/(k/(float(sum(bigram_counts[w1].values()))+k*u)))
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(mbiperplexity(i, bi_ls)))

print('\nTrigram Perplexity with Additive Smoothing')
#computes perplexity of the trigram model on a testset  
def ttriperplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for i in range(2,len(testset)):
        N += 1
        try:
            perplexity = perplexity * pow((1/model[(testset[i-2],testset[i-1])][testset[i]]),1/float(N))
        except KeyError:
            try:
                perplexity = perplexity * pow((1/(k/(k*u+float(sum(trigram_counts[(w1,w2)].values()))))),1/float(N))
            except Exception as e:
                perplexity = perplexity * pow((1/(k/(k*u))),1/float(N))
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(ttriperplexity(i, tri_ls)))


# In[33]:


# Good Turing Method affecting the lower 10 values and allocating the remanining probability in proportion of the remaining most frequent ngrams.


# In[34]:


# Good Turing Smoothing for Bigram Model
bi_gt=bigram_model(sentences)
bili=[]
co=0
for c in bigram_counts.keys():
    co+=len(bigram_counts[c])
t=0
for c in bigram_counts.keys():
        for d in bigram_counts[c].keys():
            t+=bigram_counts[c][d]
bili.append(co*co-t)
for i in range(1,12):
    temp=0
    for c in bigram_counts.keys():
        for d in bigram_counts[c].keys():
            if bigram_counts[c][d]==i:
                temp+=1
    bili.append(temp)
# To Store the modified counts
bi_mod_counts=[]
for i in range(1,len(bili)):
    bi_mod_counts.append(i*(bili[i]/float(bili[i-1])))
# Updating Counts in Model    
for c in bi_gt.keys():
        for d in bi_gt[c].keys():
            if bi_gt[c][d] in range(1,11):
                bi_gt[c][d]=bi_mod_counts[bi_gt[c][d]]
# Converting Model with Counts to Probabilities
for w1 in bi_gt:
        tot_count=float(sum(bi_gt[w1].values()))
        for w2 in bi_gt[w1]:
            bi_gt[w1][w2]=(bi_gt[w1][w2])/(tot_count)


# In[35]:


print('\nBigram test with Good Turing Smoothing\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(bi_gt[w1][w2])
        except Exception as e:
            p_val+=math.log(bi_mod_counts[0]/(float(sum(bigram_counts[w1].values()))))
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,8)))
    
    test_bigram_arr.append(p_val)

print('\nBigram Perplexity with Good Turing Smoothing')
#computes perplexity of the bigram model on a testset  
def m1biperplexity(testset, model):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            perplexity = perplexity * (1/model[testset[i-1]][testset[i]])
        except KeyError:
            perplexity = perplexity * (1/(bi_mod_counts[0]/(float(sum(bigram_counts[w1].values())))))
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(m1biperplexity(i, bi_gt)))


# In[36]:


# Good Turing Smoothing for Trigrams
tri_gt=trigram_model(sentences)
trili=[]
cou=0
for c in trigram_counts.keys():
    cou+=len(trigram_counts[c])
t=0
for c in trigram_counts.keys():
        for d in trigram_counts[c].keys():
            t+=trigram_counts[c][d]
trili.append(cou*cou-t)
for i in range(1,12):
    temp=0
    for c in trigram_counts.keys():
        for d in trigram_counts[c].keys():
            if trigram_counts[c][d]==i:
                temp+=1
    trili.append(temp)
# To store the modified counts
tri_mod_counts=[]
for i in range(1,len(trili)):
    tri_mod_counts.append(i*(trili[i]/float(trili[i-1])))
# Updating the Counts
for c in tri_gt.keys():
        for d in tri_gt[c].keys():
            if tri_gt[c][d] in range(1,11):
                tri_gt[c][d]=tri_mod_counts[tri_gt[c][d]]
# Converting Model of Counts to Probabilities
for (w1,w2) in tri_gt:
        tot_count=float(sum(tri_gt[(w1,w2)].values()))
        for w3 in tri_gt[(w1,w2)]:
            tri_gt[(w1,w2)][w3]=(tri_gt[(w1,w2)][w3])/(tot_count)


# In[37]:


test_trigram_arr=[]
print('\nTrigram test with Good Turing Smoothing\n')
for elem in test_sentences:
    p_val=0
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(tri_gt[(w1,w2)][w3])
        except Exception as e:
            try:
                p_val+=math.log((tri_mod_counts[0]/(float(sum(trigram_counts[(w1,w2)].values())))))
            except Exception as e:
                p_val+=math.log((tri_mod_counts[0]/bi_mod_counts[0]))
    print('The sequence '+ elem +' has trigram log-likelihood of '+ str(round(p_val,8)))
    
    test_trigram_arr.append(p_val)

print('\nTrigram Perplexity with Good Turing Smoothing')
#computes perplexity of the trigram model on a testset  
def t1triperplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for i in range(2,len(testset)):
        N += 1
        try:
            perplexity = perplexity * pow((1/model[(testset[i-2],testset[i-1])][testset[i]]),1/float(N))
        except KeyError:
            try:
                perplexity = perplexity * pow((1/(tri_mod_counts[0]/(float(sum(trigram_counts[(w1,w2)].values()))))),1/float(N))
            except Exception as e:
                perplexity = perplexity * pow((1/(tri_mod_counts[0]/(bi_mod_counts[0]))),1/float(N))
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(t1triperplexity(i, tri_gt)))


# In[38]:


print('Interpolation of Bigram Model with lambda value 0.2\n')

lam=0.2

test_bigram_arr=[]

for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(lam*bigram[w1][w2]+(1-lam)*unigram[w2])
        except Exception as e:
            p_val+=math.log(((1-lam)*unigram[w2]))
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,8)))
    
    test_bigram_arr.append(p_val)


# In[39]:


print('Interpolation of Bigram Model with lambda value 0.5\n')

lam=0.5

test_bigram_arr=[]

for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(lam*bigram[w1][w2]+(1-lam)*unigram[w2])
        except Exception as e:
            p_val+=math.log(((1-lam)*unigram[w2]))
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,8)))
    
    test_bigram_arr.append(p_val)


# In[40]:


print('Interpolation of Bigram Model with lambda value 0.8\n')

lam=0.8

test_bigram_arr=[]

for elem in test_sentences:
    p_val=0
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val+=math.log(lam*bigram[w1][w2]+(1-lam)*unigram[w2])
        except Exception as e:
            p_val+=math.log(((1-lam)*unigram[w2]))
    print('The sequence '+ elem +' has bigram log-likelihood of '+ str(round(p_val,8)))
    
    test_bigram_arr.append(p_val)


# In[41]:


lam=0.2
print('Bigram Perplexity with Interpolation of lambda value 0.2')
#computes perplexity of the bigram model on a testset  
def ibperplexity(testset, model1, model2):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            perplexity = perplexity * (1/((lam*model1[testset[i-1]][testset[i]])+((1-lam)*model2[testset[i]])))
        except KeyError:
            perplexity = perplexity * (1/(((1-lam)*model2[testset[i]])))
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(ibperplexity(i, bigram, unigram)))


# In[42]:


lam=0.5
print('Bigram Perplexity with Interpolation of lambda value 0.5')
#computes perplexity of the bigram model on a testset  
def ibperplexity(testset, model1, model2):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            perplexity = perplexity * (1/((lam*model1[testset[i-1]][testset[i]])+((1-lam)*model2[testset[i]])))
        except KeyError:
            perplexity = perplexity * (1/(((1-lam)*model2[testset[i]])))
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(ibperplexity(i, bigram, unigram)))


# In[43]:


lam=0.8
print('Bigram Perplexity with Interpolation of lambda value 0.8')
#computes perplexity of the bigram model on a testset  
def ibperplexity(testset, model1, model2):
    testset = testset.split()
    testset = [None] + testset + [None]
    perplexity = 1
    N = 0
    for i in range(1,len(testset)):
        N += 1
        try:
            perplexity = perplexity * (1/((lam*model1[testset[i-1]][testset[i]])+((1-lam)*model2[testset[i]])))
        except KeyError:
            perplexity = perplexity * (1/(((1-lam)*model2[testset[i]])))
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

for i in test_sentences:
    print('The sequence '+i+' has perplexity of '+str(ibperplexity(i, bigram, unigram)))


# In[44]:


sys.stdout = orig_stdout
f.close()

