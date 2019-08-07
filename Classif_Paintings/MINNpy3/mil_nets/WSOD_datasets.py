import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold

import tensorflow as tf

from tf_faster_rcnn.lib.datasets.factory import get_imdb
from IMDB_study import getDictFeaturesFasterRCNN,getTFRecordDataset
from IMDB import get_database

def load_dataset(dataset_nm='IconArt_v1',set_='train',classe=0,k_per_bag = 300,metamodel = 'FasterRCNN',demonet='res152_COCO'):
    """Load data from file, do pre-processing, split it into train/test set.
    Parameters
    -----------------
    dataset_nm : string
        Name of dataset.
    Returns
    -----------------
    datasets : list Train and test
    """
    # load data from file
    
    datasets = {}
    
    item_name,path_to_img,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC = get_database(dataset_nm)
    dict_name_file = getDictFeaturesFasterRCNN(dataset_nm,k_per_bag=k_per_bag,\
                                               metamodel=metamodel,demonet=demonet)
    
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 16
    config.inter_op_parallelism_threads = 16
    config.gpu_options.allow_growth = True
    
    ins_fea = None
    bags_full_label = None
    bags_nm = None
    bags_nm = 0 
    if set_=='train':
        name_file = dict_name_file[set_]
        if metamodel=='EdgeBoxes':
            dim_rois = 4
        else:
            dim_rois = 5
        print('num_classes',num_classes)
        next_element = getTFRecordDataset(name_file,k_per_bag =k_per_bag,\
                                          dim_rois = dim_rois,num_classes =num_classes )
        

        sess = tf.Session(config=config)
        while True:
            try:
                fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
                if ins_fea is None:
                    ins_fea = fc7s.astype(np.float32)
                    bags_full_label = labels.astype(np.float32)
                    bags_nm = name_imgs
                else:
                    ins_fea = np.vstack((ins_fea,fc7s)).astype(np.float32)
                    bags_full_label = np.vstack((bags_full_label,labels)).astype(np.float32)
                    bags_nm = np.concatenate((bags_nm,name_imgs))
                #for k in range(len(labels)):
                    #name_im = name_imgs[k].decode("utf-8")
                    
            except tf.errors.OutOfRangeError:
                break
    
        sess.close()
        
    bags_full_label = np.array(bags_full_label)
    bags_label = bags_full_label[:,classe] 

#    data = sio.loadmat('./dataset/'+dataset_nm+'.mat')
#    ins_fea = data['x']['data'][0,0]
#    if dataset_nm.startswith('musk'):
#        bags_nm = data['x']['ident'][0,0]['milbag'][0,0]
#    else:
#        bags_nm = data['x']['ident'][0,0]['milbag'][0,0][:,0]
#    bags_label = data['x']['nlab'][0,0][:,0] - 1

    # L2 norm for musk1 and musk2

    if dataset_nm.startswith('newsgroups') is False:
        mean_fea = np.mean(ins_fea, axis=(0,1), keepdims=True)+1e-6
        std_fea = np.std(ins_fea, axis=(0,1), keepdims=True)+1e-6
        ins_fea = np.divide(ins_fea-mean_fea, std_fea)
    else:
        mean_fea = np.ones((1,1,ins_fea.shape[2]))
        std_fea = np.ones((1,1,ins_fea.shape[2]))
        
    bags_fea = []
    for id, bag_nm in enumerate(bags_nm):
        bag_fea = ([], [])
        for ins_idx in range(k_per_bag):
            bag_fea[0].append(ins_fea[id,ins_idx,:])
            bag_fea[1].append(bags_label[id])
        bags_fea.append(bag_fea)
#        
#    # store data in bag level
#    ins_idx_of_input = {}            # store instance index of input
#    for id, bag_nm in enumerate(bags_nm):
#        if bag_nm in ins_idx_of_input: ins_idx_of_input[bag_nm].append(id)
#        else: ins_idx_of_input[bag_nm] = [id]
#    bags_fea = []
#    for bag_nm, ins_idxs in list(ins_idx_of_input.items()):
#        bag_fea = ([], [])
#        for ins_idx in ins_idxs:
#            print('ins_fea[ins_idx][0]',ins_fea[ins_idx][0].shape)
#            bag_fea[0].append(ins_fea[ins_idx][0])
#            bag_fea[1].append(bags_label[ins_idx])
#        bags_fea.append(bag_fea)

    datasets['train']= bags_fea
    

#    # random select 90% bags as train, others as test
#    num_bag = len(bags_fea)
#    kf = KFold(num_bag, n_folds=n_folds, shuffle=True, random_state=None)
#    datasets = []
#    for train_idx, test_idx in kf:
#        
#        dataset['train'] = [bags_fea[ibag] for ibag in train_idx]
#        dataset['test'] = [bags_fea[ibag] for ibag in test_idx]
#        datasets.append(dataset)
    return datasets,bags_full_label,mean_fea,std_fea
