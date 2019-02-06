# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:48:55 2018

@author: gonthier
"""
import glob
import numpy as np
import os

def list_tofloat(l):
    r = []
    for elt in l:
        r += [float(elt)]
    return(r)

def get_Set(Data,index):
    """
    Provide the element in Data from the index
    """
    DataS = [ np.array(Data[i]) for i in index]
    #DataS = DataS[0]
    return(DataS)

def ExtractSubsampledSIVAL(verbose=False):
    """
    The goal of this function is too extract a sample of the SIVAL dataset as 
    it is done in the survey Carbonneau 2016
    
    For each class the 60 positives bag are selected and 5 images/bag from each 
    of 24  other class are randomly selected
    
    We do it 5 resamples
    """
    
    
    Dataset = ExtractSIVAL()
    list_names,bags,labels_bags,labels_instance = Dataset
    
    # number of differents sets
    num_sample = 5
    num_of_images_per_samples = 5
    list_pos_bag = []
    for i,classe_name in enumerate(list_names):
        labels_bags_for_c = labels_bags[i]
        pos_index_c = np.where(labels_bags_for_c==1) #Select the positive bag
        list_pos_bag += [pos_index_c]
    
    class_with_59posElt = []
    for i in range(len(list_names)):
        if len(list_pos_bag[i][0])==59:
            if verbose: print('In SIVAL',list_names[i],' only have 59 positives case')
            class_with_59posElt += [i]
        
        
    bags_sampled = [[[] for k in range(num_sample)] for i in range(len(list_names))]
    labels_bags_sampled = [[[] for k in range(num_sample)] for i in range(len(list_names))]
    labels_instance_sampled = [[[] for k in range(num_sample)] for i in range(len(list_names))]
    for i,_ in enumerate(list_names):
        pos_index_class_interest = list_pos_bag[i][0]
        list_of_index = [[] for k in range(num_sample)]
        for k in range(num_sample):
            list_of_index[k] = pos_index_class_interest
        for j,_ in enumerate(list_names): 
            if not(j==i):
                pos_index_c = list_pos_bag[j]
                sample_5 = np.random.choice(pos_index_c[0],num_sample*num_of_images_per_samples,replace=False)
                for k in range(num_sample):
                    samples_index = sample_5[k*num_of_images_per_samples:(k+1)*num_of_images_per_samples]
                    list_of_index[k] = np.hstack((list_of_index[k],samples_index))
#                    print("len(get_Set(bags,samples_index))",len(get_Set(bags,samples_index)))
#                    bags_sampled[i][k] += [get_Set(bags,samples_index)]
#                    print("len(bags_sampled[i][k])",len(bags_sampled[i][k]))
#                    labels_bags_sampled[i][k] += [get_Set(labels_bags[i],samples_index)] 
#                    labels_instance_sampled[i][k] += [get_Set(labels_instance[i],samples_index)] 
        #            => Cela ne marche pas du tout :(
        for k in range(num_sample):
            if not(i in class_with_59posElt):
                assert(len(list_of_index[k])==60+num_of_images_per_samples*24)
            else:
                assert(len(list_of_index[k])==59+num_of_images_per_samples*24)
            bags_sampled[i][k] = get_Set(bags,list_of_index[k])
            labels_bags_sampled[i][k] = get_Set(labels_bags[i],list_of_index[k])
            labels_instance_sampled[i][k] = get_Set(labels_instance[i],list_of_index[k])
            
            
    for i,_ in enumerate(list_names):
        for k in range(num_sample):
            if not(i in class_with_59posElt):
                assert(len(bags_sampled[i][k])==60+num_of_images_per_samples*24)
                assert(len(labels_bags_sampled[i][k])==60+num_of_images_per_samples*24)
                assert(len(labels_instance_sampled[i][k])==60+num_of_images_per_samples*24)
            else:
                assert(len(bags_sampled[i][k])==59+num_of_images_per_samples*24)
                assert(len(labels_bags_sampled[i][k])==59+num_of_images_per_samples*24)
                assert(len(labels_instance_sampled[i][k])==59+num_of_images_per_samples*24)
      
    Dataset_sampled = list_names,bags_sampled,labels_bags_sampled,labels_instance_sampled
    
    return(Dataset_sampled)  
    
def ExtractSIVAL():
    """
    This return a dictionnary in which each element is a list of the exemple of 
    the class
    
    
    This function return a list of the 
    
    """
    number_of_class = 25
    number_of_bag = 1499
    bags = [] # List of the features
    list_names = []
    path_directory = 'SIVAL'
    script_dir = os.path.dirname(__file__) 
    data_to_read = os.path.join(script_dir,path_directory,'*.data')
    allclasses=glob.glob(data_to_read)
    
    labels_instance = [[[] for i in range(number_of_bag)] for j in range(number_of_class)]   # List of the labels of the instance
    labels_bags = [np.empty((number_of_bag,)) for j in range(number_of_class)] 
    
    list_names_bags = []
    FirstTimeBag = True
    bag_id_int = -1
    for c,classe_name in enumerate(allclasses):
        classe = os.path.split(classe_name)[-1]
        elt_name = classe.split('.')[0]
        list_names += [elt_name]
        with open(classe_name,'r') as f:
            content = f.readlines()
            bag_id_old = -1
            bag = None
#            labels_instance_c = []
#            labels = []
            for line in content:
                line_splitted=line.replace('\n','').split(',')
                bag_id = str(line_splitted[0])
                if bag_id in list_names_bags:
                    bag_id_int = np.where(np.array(list_names_bags)==bag_id)[0][0]
                else:
                    list_names_bags += [bag_id]
                    bag_id_int += 1
#                instance_id = int(line_splitted[1]) # Instance ID inside the bag !
                instance_label = float(line_splitted[-1])
                features = np.array(list_tofloat(line_splitted[2:-1]))
                if elt_name in bag_id.lower():
                    bag_label = 1
                else:
                    bag_label = -1
#                print('bag_id_int',bag_id_int)
                labels_bags[c][bag_id_int] = bag_label
                labels_instance[c][bag_id_int] += [2*instance_label-1] 
                
                
                if FirstTimeBag:
                    if bag_id_old==bag_id_int:
                        bag = np.vstack((bag,features))
                    else:
                        if not(bag_id_old==-1):
                            bags += [bag]
                        bag_id_old = bag_id_int
                        bag = features
            # End of reading the file
            
        # End of the class
        if FirstTimeBag:
            bags += [bag]
            FirstTimeBag = False
#        labels_instance += [np.array(labels_instance_c)]
#        labels_bags  += [np.array(labels)]

    for j in range(number_of_class):
        for i in range(number_of_bag):
            labels_instance_c_b = labels_instance[j][i]
            labels_instance[j][i] = np.array(labels_instance_c_b)
            assert(np.max(labels_instance[j][i])==labels_bags[j][i])
            assert(len(bags[i])==len(labels_instance[j][i]))

    
    # Quick test
    assert(len(bags)==number_of_bag)
    for j in range(number_of_class):
        assert(len(labels_instance[j])==number_of_bag)
        assert(len(labels_instance[j])==number_of_bag)
        assert(len(labels_bags[j])==number_of_bag)
        assert(len(labels_bags[j])==number_of_bag)
    
    Dataset = list_names,bags,labels_bags,labels_instance
            
    return(Dataset)
  
if __name__=='__main__':
    ExtractSubsampledSIVAL(verbose=True)