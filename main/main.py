'''
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
import os
from corpus.Discourse import *

from GrammarML import GrammarCluster

corpus_dir = "/media/DADOS/UNICAMP/Summ-it_v3.0/corpusAnotado_CCR"

discourses = []        
alg_options = {'bind_const': False, 'utt_type':'sentence', 'veins_head':'no'}

load_discourses = False

cluster = GrammarCluster.GrammarCluster(discourses, "/media/DADOS/UNICAMP/ml_parser_result.csv")

if load_discourses:
    for dis_addr in os.listdir(corpus_dir):
           
                
        discourse = Discourse(None, alg_options)
        print "Loading corpus %s ...\n" % dis_addr   
        discourse.loadFromSummitCorpus(corpus_dir + "/"+ dis_addr)
           
        discourses.append(discourse)
    
    print "Corpora loaded, registering samples...\n"
    cluster.readSampplesFromDiscourse()
    
    print "Saving samples into files\n"
    cluster.exportSamples("/media/DADOS/UNICAMP/ml_parser_sample.csv")
    cluster.saveSamplesToNdarrayFile("/media/DADOS/UNICAMP/ml_parser_sample.ndarray")
else:
    print "Skip loading corpora\n"
        

print "Clustering\n"
cluster.doClusterFromNdarrayFile("/media/DADOS/UNICAMP/ml_parser_sample.ndarray.npy")
cluster.start()
    