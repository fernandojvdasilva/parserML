'''
Created on 01/05/2013

@author: fernando

    Copyright 2009-2013 Fernando J. V. da Silva
    
    This file is part of centering_py.

    centering_py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    centering_py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with centering_py.  If not, see <http://www.gnu.org/licenses/>.

'''

from corpus import Discourse
import Levenshtein
import os
from sklearn.cluster import AffinityPropagation, DBSCAN
from threading import Thread
import numpy as np

class GrammarClusterSample(object):    
    SAMPLE_ATTR_FREQ = 0
    SAMPLE_ATTR_STRLEN = 1
    # Number of words in which the Levenshtein distance is smaller or equal than: 
    # half of the string length + (the bigger string length - the smaller string length) 
    SAMPLE_ATTR_NUMSIMILAR = 2
    # Levenshtein distance average for the strings satisfying the condition above       
    SAMPLE_ATTR_AVG_LEV_DISTANCE = 3
    SAMPLE_ATTR_MIN_DISTANCE_TO_BEGIN_SENTENCE = 4
    SAMPLE_ATTR_MIN_DISTANCE_TO_END_SENTENCE = 5
    SAMPLE_ATTR_AVG_DISTANCE_TO_BEGIN_SENTENCE = 6
    
    
    def __init__(self):
        self.attributes = [0, 0, 0, 0, -1, -1, 0]
        self.word = ''
        self.distances_begin = []
        self.lev_distances = []

    def __getitem__(self, key):
        if key == 'word':
            return self.word
    
    def wordsAreSimilar(self, word1, word2, dist):
#        word1Len = len(word1)
#        word2Len = len(word2)
#        difference = 0
#        if word1Len > word2Len:
#            difference = word1Len - word2Len
#            biggerLen = word1Len
#        else:
#            difference = word2Len - word1Len
#            biggerLen = word2Len
#            
#        if dist < (difference + (2 * (biggerLen / 3))):
#            return True
#        else:
#            return False
        if len(word1) > 3 and dist < 3:
            return True
        else:
            return False
            
             
    
    def registerSample(self, word, samples):
        self.word = word
        self.attributes[self.SAMPLE_ATTR_FREQ] += 1                    
        sen = word.sentence
        distance_begin = 0
        distance_end = 0
        word_found = False
        for w in sen.words:
            if w == word:
                word_found = True
            if not word_found:
                distance_begin += 1
            else:
                distance_end += 1
        if self.attributes[self.SAMPLE_ATTR_MIN_DISTANCE_TO_BEGIN_SENTENCE] == -1 or \
           self.attributes[self.SAMPLE_ATTR_MIN_DISTANCE_TO_BEGIN_SENTENCE] > distance_begin:
            self.attributes[self.SAMPLE_ATTR_MIN_DISTANCE_TO_BEGIN_SENTENCE] = distance_begin
            
        if self.attributes[self.SAMPLE_ATTR_MIN_DISTANCE_TO_END_SENTENCE] == -1 or \
           self.attributes[self.SAMPLE_ATTR_MIN_DISTANCE_TO_END_SENTENCE] > distance_end:
            self.attributes[self.SAMPLE_ATTR_MIN_DISTANCE_TO_END_SENTENCE] = distance_end
        
        self.distances_begin.append(distance_begin)
        self.attributes[self.SAMPLE_ATTR_AVG_DISTANCE_TO_BEGIN_SENTENCE] = \
            sum(self.distances_begin) / len(self.distances_begin)
        
        if 1 == self.attributes[self.SAMPLE_ATTR_FREQ]:
            self.attributes[self.SAMPLE_ATTR_STRLEN] = len(word.properties['text'])            
                    
    def toString(self):
        str = '%s,%d,%d,%d,%d,%d,%d,%d\n' % (self.word.properties['text'],\
                                           self.attributes[self.SAMPLE_ATTR_FREQ],\
                                           self.attributes[self.SAMPLE_ATTR_STRLEN],\
                                           self.attributes[self.SAMPLE_ATTR_NUMSIMILAR],\
                                           self.attributes[self.SAMPLE_ATTR_AVG_LEV_DISTANCE],\
                                           self.attributes[self.SAMPLE_ATTR_MIN_DISTANCE_TO_BEGIN_SENTENCE],\
                                           self.attributes[self.SAMPLE_ATTR_MIN_DISTANCE_TO_END_SENTENCE],\
                                           self.attributes[self.SAMPLE_ATTR_AVG_DISTANCE_TO_BEGIN_SENTENCE])
        return str
            
class GrammarCluster(Thread):

    def __init__(self, discourses, result_file_path):
        self.discourses = discourses
        self.samples = []
        self.csv_samples_path = "/media/DADOS/UNICAMP/ml_parser_sample.csv"
        self.result_file_path = result_file_path
        self.af = AffinityPropagation(max_iter=10,copy=False,verbose=True,affinity='euclidean')
        Thread.__init__(self)        
    
    def findSample(self, word):
        for sample in self.samples:
            if sample['word'].properties['text'] == word.properties['text']:
                return sample
        return None
    
    def getSamplesArray(self):
        data = []
        for sample in self.samples:
            data.append(sample.attributes)
        return data
    
    def readSampplesFromDiscourse(self):
        for dis in self.discourses:
            for sen in dis.sentences:
                for word in sen.words:
                   sample = self.findSample(word)
                   if None == sample:
                       sample = GrammarClusterSample()
                       sample.registerSample(word, self.samples)
                       self.samples.append(sample)
                   else:
                       sample.registerSample(word, self.samples)  
        
        for sample in self.samples:
            for sample2 in self.samples:
                if sample == sample2:
                    continue
                dist = Levenshtein.distance(sample.word.properties['text'], sample2.word.properties['text'])
                
                if sample.wordsAreSimilar(sample.word.properties['text'], sample2.word.properties['text'], dist):
                    sample.attributes[sample.SAMPLE_ATTR_NUMSIMILAR] += 1 
                    sample.lev_distances.append(dist)
                    if len(sample.lev_distances) > 0:
                        sample.attributes[sample.SAMPLE_ATTR_AVG_LEV_DISTANCE] = \
                                        sum(sample.lev_distances) / len(sample.lev_distances)
                    else:
                        sample.attributes[sample.SAMPLE_ATTR_AVG_LEV_DISTANCE] = 0
    
    def saveSamplesToNdarrayFile(self, file_path):
        np.save(file_path, self.getSamplesArray())        
        
    def doClusterFromNdarrayFile(self, file_path):
        ndarray = np.load(file_path)
        self.data = ndarray
    
    def doCluster(self, result_file_path):    
        # We use Affinity Propagation            
        self.af.fit(self.data)        
        self.labels = af.labels_        
        #db = DBSCAN().fit(self.data, eps=0.2)
        #self.sample_labels = db.core_sample_indices_
        #self.labels = db.labels_
        print "Clusters Results:\n"
        #print "Number of clustered samples: %d \n" % (len(self.sample_labels) - (1 if -1 in self.sample_labels else 0))
        
        csv_file = open(self.csv_samples_path, 'r')
        result_file = open(result_file_path, 'w')        
        for line, label in zip(csv_file, self.labels):
            fields = line.split(',')
            if label >= 0:
                result_file.write('%s,%d\n' % (fields[0], label))
        result_file.close()    
        csv_file.close()                

    def run(self):
        self.doCluster(self.result_file_path)
        
    def exportSamples(self, file_path):    
        self.csv_samples_path = file_path
        sample_str = ''    
        for sample in self.samples:
            sample_str += sample.toString()
        csv_file = open(file_path, 'w')
        csv_file.write(sample_str)
        csv_file.close
        