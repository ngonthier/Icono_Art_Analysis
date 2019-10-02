#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:14:19 2019

In this script we are looking at the transfer learning of a VGG with some 
statistics imposed on the features maps of the layers

@author: gonthier
"""

from trouver_classes_parmi_K import TrainClassif
import numpy as np
import os.path
from Study_Var_FeaturesMaps import get_dict_stats,numeral_layers_index
from Stats_Fcts import vgg_cut,vgg_InNorm_adaptative,vgg_InNorm,vgg_BaseNorm,\
    load_crop_and_process_img,VGG_baseline_model,vgg_AdaIn,ResNet_baseline_model
from IMDB import get_database
import pickle
import pathlib
from Classifier_On_Features import TrainClassifierOnAllClass,PredictOnTestSet
from sklearn.metrics import average_precision_score,recall_score,make_scorer,\
    precision_score,label_ranking_average_precision_score,classification_report
from sklearn.metrics import matthews_corrcoef,f1_score
from sklearn.preprocessing import StandardScaler
from Custom_Metrics import ranking_precision_score
from LatexOuput import arrayToLatex
from sklearn.model_selection import train_test_split
import tensorflow as tf

def compute_ref_stats(dico,style_layers,type_ref='mean',imageUsed='all',whatToload = 'varmean',applySqrtOnVar=False):
    """
    This function compute a reference statistics on the statistics of the whole dataset
    """
    vgg_stats_values = []
    for l,layer in enumerate(style_layers):
        stats = dico[layer]
        if whatToload == 'varmean':
            if imageUsed=='all':
                if type_ref=='mean':
                    mean_stats = np.mean(stats,axis=0) # First colunm = variance, second = mean
                    mean_of_means = mean_stats[1,:]
                    mean_of_vars = mean_stats[0,:]
                    if applySqrtOnVar:
                        mean_of_vars = np.sqrt(mean_of_vars) # STD
                    vgg_stats_values += [[mean_of_means,mean_of_vars]]
                    # To return vgg_mean_vars_values
    return(vgg_stats_values)

def get_dict_stats_BaseNormCoherent(target_dataset,source_dataset,target_number_im_considered,\
                                    style_layers,\
                                    list_mean_and_std_source,whatToload,saveformat='h5',\
                                    getBeforeReLU=False,target_set='trainval',applySqrtOnVar=True):
    """
    The goal of this function is to compute a version of the statistics of the 
    features of the VGG 
    """
    dict_stats_coherent = {} 
    for i_layer,layer_name in enumerate(style_layers):
        #print(i_layer,layer_name)
        if i_layer==0:
            style_layers_firstLayer = [layer_name]
            dict_stats_target0 = get_dict_stats(target_dataset,target_number_im_considered,\
                                                style_layers_firstLayer,\
                                                whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,\
                                                set=target_set)
            dict_stats_coherent[layer_name] = dict_stats_target0[layer_name]
            list_mean_and_std_target_i_m1 = compute_ref_stats(dict_stats_target0,\
                                            style_layers_firstLayer,type_ref='mean',\
                                            imageUsed='all',whatToload=whatToload,\
                                            applySqrtOnVar=applySqrtOnVar)
            current_list_mean_and_std_target = list_mean_and_std_target_i_m1
        else:
            list_mean_and_std_source_i = list_mean_and_std_source[0:i_layer]
            style_layers_imposed = style_layers[0:i_layer]
            style_layers_exported = [style_layers[i_layer]]
            list_mean_and_std_source_i = list_mean_and_std_source[0:i_layer]
            
            dict_stats_target_i = get_dict_stats(target_dataset,target_number_im_considered,\
                                                 style_layers=style_layers_exported,whatToload=whatToload,\
                                                 saveformat='h5',getBeforeReLU=getBeforeReLU,\
                                                 set=target_set,Net='VGGBaseNormCoherent',\
                                                 style_layers_imposed=style_layers_imposed,\
                                                 list_mean_and_std_source=list_mean_and_std_source_i,\
                                                 list_mean_and_std_target=current_list_mean_and_std_target)
            dict_stats_coherent[layer_name] = dict_stats_target_i[layer_name]
            # Compute the next statistics 
            list_mean_and_std_target_i = compute_ref_stats(dict_stats_target_i,\
                                            style_layers_exported,type_ref='mean',\
                                            imageUsed='all',whatToload=whatToload,\
                                            applySqrtOnVar=applySqrtOnVar)
            current_list_mean_and_std_target += [list_mean_and_std_target_i[-1]]
            
    return(dict_stats_coherent,current_list_mean_and_std_target)

def learn_and_eval(target_dataset,source_dataset='ImageNet',final_clf='LinearSVC',features='fc2',constrNet='VGG',kind_method='FT',
                   style_layers = ['block1_conv1',
                                    'block2_conv1',
                                    'block3_conv1', 
                                    'block4_conv1', 
                                    'block5_conv1'
                                   ],normalisation=False,gridSearch=True,ReDo=False,\
                                   transformOnFinalLayer='',number_im_considered = 1000,\
                                   set='',getBeforeReLU=False,forLatex=False,epochs=20,\
                                   pretrainingModif=True,weights='imagenet',opt_option=[0.01],\
                                   optimizer='adam',freezingType='FromTop'):
    """
    @param : the target_dataset used to train classifier and evaluation
    @param : source_dataset : used to compute statistics we will imposed later
    @param : final_clf : the final classifier can be
    TODO : linear SVM - MLP - perceptron - MLP one per class - MLP with finetuning of the net
    @param : features : which features we will use
    TODO : fc2, fc1, max spatial, max et min spatial
    @param : constrNet the constrained net used
    TODO : VGGInNorm, VGGInNormAdapt seulement sur les features qui répondent trop fort, VGGGram
    @param : kind_method the type of methods we will use : TL or FT
    @param : if we use a set to compute the statistics
    @param : getBeforeReLU=False if True we will impose the statistics before the activation ReLU fct
    @param : forLatex : only plot performance score to print them in latex
    @param : epochs number of epochs for the finetuning (FT case)
    @param : pretrainingModif : we modify the pretrained net for the case FT + VGG 
        it can be a boolean True of False or a 
    @param : opt_option : learning rate different for the SGD
    @param : freezingType : the way we unfreeze the pretained network : 'FromBottom','FromTop','Alter'
    """
    assert(freezingType in ['FromBottom','FromTop','Alter'])
    
    output_path = os.path.join(os.sep,'media','gonthier','HDD2','output_exp','Covdata',target_dataset)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    if kind_method=='FT':
        model_output_path = os.path.join(output_path,'model')
        pathlib.Path(model_output_path).mkdir(parents=True, exist_ok=True) 
    
    num_layers = numeral_layers_index(style_layers)
    # Compute statistics on the source_dataset
    if source_dataset is None:
        constrNet='VGG'
        
    # Load info about dataset
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,\
    path_data,Not_on_NicolasPC = get_database(target_dataset)
        
    name_base = constrNet + '_'  +target_dataset +'_'
    if not(constrNet=='VGG'):
        if kind_method=='TL':
            name_base += source_dataset +str(number_im_considered)
        name_base +=  '_' + num_layers
    if kind_method=='FT' and (weights is None):
        name_base += '_RandInit' # Random initialisation 
    if kind_method=='FT' and not(optimizer=='adam'):
        name_base += '_'+optimizer
        if len(opt_option)==2:
            multiply_lrp, lr = opt_option
            name_base += '_lrp'+str(multiply_lrp)+'_lr'+str(lr)
        if len(opt_option)==1:
            lr = opt_option[0]
            name_base += '_lr'+str(lr)
    if not(set=='' or set is None):
        name_base += '_'+set
    if constrNet=='VGG':
        getBeforeReLU  = False
    if constrNet in ['VGG','ResNet50'] and kind_method=='FT':
        if type(pretrainingModif)==bool:
            if pretrainingModif==False:
                name_base +=  '_wholePretrainedNetFreeze'
        else:
            if not(freezingType=='FromTop'):
                name_base += '_'+freezingType
            name_base += '_unfreeze'+str(pretrainingModif)
            
    if getBeforeReLU:
        name_base += '_BeforeReLU'
    name_base +=  features 
    if not((transformOnFinalLayer is None) or (transformOnFinalLayer=='')):
       name_base += '_'+ transformOnFinalLayer
    name_base += '_' + kind_method   
    
    # features can be 'flatten' with will output a 25088 dimension vectors = 7*7*512 features
    
    if kind_method=='TL': # Transfert Learning
        final_layer = features
        name_pkl_im = target_dataset +'.pkl'
        name_pkl_values  = name_base+ '_Features.pkl'
        name_pkl_im = os.path.join(output_path,name_pkl_im)
        name_pkl_values = os.path.join(output_path,name_pkl_values)
        
        if not os.path.isfile(name_pkl_values):
            if not(forLatex):
                print('== We will compute the reference statistics ==')
                print('Saved in ',name_pkl_values)
            features_net = None
            im_net = []
            # Load Network 
            if constrNet=='VGG':
                network_features_extraction = vgg_cut(final_layer,\
                                                      transformOnFinalLayer=transformOnFinalLayer)
            elif constrNet=='VGGInNorm' or constrNet=='VGGInNormAdapt':
                whatToload = 'varmean'
                dict_stats = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                       whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=set)
                # Compute the reference statistics
                vgg_mean_stds_values = compute_ref_stats(dict_stats,style_layers,type_ref='mean',\
                                                     imageUsed='all',whatToload =whatToload,
                                                     applySqrtOnVar=True)
                if constrNet=='VGGInNormAdapt':
                    network_features_extraction = vgg_InNorm_adaptative(style_layers,
                                                                       vgg_mean_stds_values,
                                                                       final_layer=final_layer,
                                                                       transformOnFinalLayer=transformOnFinalLayer,
                                                                       HomeMadeBatchNorm=True,getBeforeReLU=getBeforeReLU)
                elif constrNet=='VGGInNorm':
                    network_features_extraction = vgg_InNorm(style_layers,
                                                            vgg_mean_stds_values,
                                                            final_layer=final_layer,
                                                            transformOnFinalLayer=transformOnFinalLayer,
                                                            HomeMadeBatchNorm=True,getBeforeReLU=getBeforeReLU)
            elif constrNet=='VGGBaseNorm':
                whatToload = 'varmean'
                dict_stats_source = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                       whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=set)
                # Compute the reference statistics
                list_mean_and_std_source = compute_ref_stats(dict_stats_source,style_layers,type_ref='mean',\
                                                     imageUsed='all',whatToload =whatToload,
                                                     applySqrtOnVar=True)
                target_number_im_considered = None
                target_set = 'trainval' # Todo ici
                dict_stats_target = get_dict_stats(target_dataset,target_number_im_considered,style_layers,\
                       whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=target_set)
                # Compute the reference statistics
                list_mean_and_std_target = compute_ref_stats(dict_stats_target,style_layers,type_ref='mean',\
                                                     imageUsed='all',whatToload =whatToload,
                                                     applySqrtOnVar=True)
#                
#                
                network_features_extraction = vgg_BaseNorm(style_layers,list_mean_and_std_source,
                    list_mean_and_std_target,final_layer=final_layer,transformOnFinalLayer=transformOnFinalLayer,
                    getBeforeReLU=getBeforeReLU)
                
            elif constrNet=='VGGBaseNormCoherent':
                # A more coherent way to compute the VGGBaseNormalisation
                # We will pass the dataset several time through the net modify bit after bit to 
                # get the coherent mean and std of the target domain
                whatToload = 'varmean'
                dict_stats_source = get_dict_stats(source_dataset,number_im_considered,style_layers,\
                       whatToload,saveformat='h5',getBeforeReLU=getBeforeReLU,set=set)
                # Compute the reference statistics
                list_mean_and_std_source = compute_ref_stats(dict_stats_source,style_layers,type_ref='mean',\
                                                     imageUsed='all',whatToload =whatToload,
                                                     applySqrtOnVar=True)
                target_number_im_considered = None
                target_set = 'trainval'
                dict_stats_target,list_mean_and_std_target = get_dict_stats_BaseNormCoherent(target_dataset,source_dataset,target_number_im_considered,\
                       style_layers,list_mean_and_std_source,whatToload,saveformat='h5',\
                       getBeforeReLU=getBeforeReLU,target_set=target_set,\
                       applySqrtOnVar=True) # It also computes the reference statistics (mean,var)
                
                network_features_extraction = vgg_BaseNorm(style_layers,list_mean_and_std_source,
                    list_mean_and_std_target,final_layer=final_layer,transformOnFinalLayer=transformOnFinalLayer,
                    getBeforeReLU=getBeforeReLU)
            
            else:
                raise(NotImplementedError)
            if not(forLatex):
                print('== We will compute the bottleneck features ==')
            # Compute bottleneck features on the target dataset
            for i,name_img in  enumerate(df_label[item_name]):
                im_path =  os.path.join(path_to_img,name_img+'.jpg')
                image = load_crop_and_process_img(im_path)
                features_im = network_features_extraction.predict(image)
                if features_net is not None:
                    features_net = np.vstack((features_net,np.array(features_im)))
                    im_net += [name_img]
                else:
                    features_net =np.array(features_im)
                    im_net = [name_img]
            with open(name_pkl_values, 'wb') as pkl:
                pickle.dump(features_net,pkl)
            
            with open(name_pkl_im, 'wb') as pkl:
                pickle.dump(im_net,pkl)
        else: # Load the precomputed data
            with open(name_pkl_values, 'rb') as pkl:
                features_net = pickle.load(pkl)
            
            with open(name_pkl_im, 'rb') as pkl:
                im_net = pickle.load(pkl)
    
    AP_file  = name_base+'_'+final_clf
    if kind_method=='TL':
        if normalisation:
            AP_file +=  '_Norm' 
            if gridSearch:
                AP_file += '_GS'
    elif kind_method=='FT':
       AP_file +=  '_'+str(epochs)
    
    AP_file_base =  AP_file
    AP_file_pkl =AP_file_base+'_AP.pkl'
    APfilePath =  os.path.join(output_path,AP_file_pkl)
    
    if kind_method=='TL':
        Latex_str = constrNet 
        if style_layers==['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1']:
            Latex_str += ' Block1-5\_conv1'
        elif style_layers==['block1_conv1','block2_conv1']:
            Latex_str += ' Block1-2\_conv1' 
        elif style_layers==['block1_conv1']:
            Latex_str += ' Block1\_conv1' 
        else:
            for layer in style_layers:
                Latex_str += layer
        Latex_str += ' ' +features.replace('_','\_')
        Latex_str += ' ' + transformOnFinalLayer
        Latex_str += ' '+final_clf
        if gridSearch:
            Latex_str += 'GS'
        else:
            Latex_str += ' no GS'
        if getBeforeReLU:
            Latex_str += ' BFReLU'
    elif kind_method=='FT':
        Latex_str = constrNet +' '+transformOnFinalLayer  + ' ep :' +str(epochs)
        if type(pretrainingModif)==bool:
            if not(pretrainingModif):
                Latex_str += ' All Freeze'
        else:
            Latex_str += ' Unfreeze '+str(pretrainingModif) + ' ' +freezingType
        if getBeforeReLU:
            Latex_str += ' BFReLU'
        if weights is None:
            Latex_str += ' RandInit'
    
    if not(os.path.isfile(APfilePath)) or ReDo:
        
        if target_dataset=='Paintings':
            sLength = len(df_label[item_name])
            classes_vectors = np.zeros((sLength,num_classes))
            for i in range(sLength):
                for j in range(num_classes):
                    if( classes[j] in df_label['classe'][i]):
                        classes_vectors[i,j] = 1
            if kind_method=='FT':
                df_copy = df_label.copy()
                for j,c in enumerate(classes):
                    df_copy[c] = classes_vectors[:,j].astype(int)
                    #df_copy[c] = df_copy[c].apply(lambda x : bool(x))
                df_label = df_copy
                df_label_test = df_label[df_label['set']=='test']
                y_test = classes_vectors[df_label['set']=='test',:]
        else:
            raise(NotImplementedError)
    
        if kind_method=='TL':
            # Get Train set in form of numpy array
            index_train = df_label['set']=='train'
            if not(forLatex):
                print('classes_vectors.shape',classes_vectors.shape)
                print('features_net.shape',features_net.shape)
            X_train = features_net[index_train,:]
            y_train = classes_vectors[df_label['set']=='train',:]
            
            X_test= features_net[df_label['set']=='test',:]
            
            X_val = features_net[df_label['set']==str_val,:]
            y_val = classes_vectors[df_label['set']==str_val,:]
            
            Xtrainval = np.vstack([X_train,X_val])
            ytrainval = np.vstack([y_train,y_val])
            
            if normalisation:
                scaler = StandardScaler()
                Xtrainval = scaler.fit_transform(Xtrainval)
                X_test = scaler.transform(X_test)
            
            dico_clf=TrainClassifierOnAllClass(Xtrainval,ytrainval,clf=final_clf,gridSearch=gridSearch)
            # Prediction
            dico_pred = PredictOnTestSet(X_test,dico_clf,clf=final_clf)
            metrics = evaluationScoreDict(y_test,dico_pred)
            
        elif kind_method=='FT':
            # We fineTune a VGG
            if constrNet=='VGG':
                getBeforeReLU = False
                model = VGG_baseline_model(num_of_classes=num_classes,pretrainingModif=pretrainingModif,
                                           transformOnFinalLayer=transformOnFinalLayer,weights=weights,
                                           optimizer=optimizer,opt_option=opt_option,freezingType=freezingType)
            elif constrNet=='ResNet50':
                getBeforeReLU = False
                model = ResNet_baseline_model(num_of_classes=num_classes,pretrainingModif=pretrainingModif,
                                           transformOnFinalLayer=transformOnFinalLayer,weights=weights,\
                                           res_num_layers=50)
                
            elif constrNet=='VGGAdaIn':
                model = vgg_AdaIn(style_layers,num_of_classes=num_classes,weights=weights,
                          transformOnFinalLayer=transformOnFinalLayer,getBeforeReLU=getBeforeReLU)
            else:
                print(constrNet,'is unkwon in the context of TL')
                raise(NotImplementedError)
            
            model = FineTuneModel(model,dataset=target_dataset,df=df_label,\
                                    x_col=item_name,y_col=classes,path_im=path_to_img,\
                                    str_val=str_val,num_classes=len(classes),epochs=epochs,\
                                    Net=constrNet)
            model_path = os.path.join(model_output_path,AP_file_base+'.h5')
            model.save(model_path,include_optimizer=True)
            # Prediction
            predictions = predictionFT_net(model,df_test=df_label_test,x_col=item_name,\
                                           y_col=classes,path_im=path_to_img,Net=constrNet)
            print(predictions.shape)
            print(y_test.shape)
            metrics = evaluationScore(y_test,predictions)    
            
        with open(APfilePath, 'wb') as pkl:
            pickle.dump(metrics,pkl)
                
            
    else:

        with open(APfilePath, 'rb') as pkl:
            metrics = pickle.load(pkl)
    AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class = metrics
    
    if not(forLatex):
        print(target_dataset,source_dataset,number_im_considered,final_clf,features,transformOnFinalLayer,\
              constrNet,kind_method,'GS',gridSearch,'norm',normalisation,'getBeforeReLU',getBeforeReLU)
        print(style_layers)
    
    
#    print(Latex_str)
    #VGGInNorm Block1-5\_conv1 fc2 LinearSVC no GS BFReLU
    str_part2 = arrayToLatex(AP_per_class,per=True)
    Latex_str += str_part2
    Latex_str = Latex_str.replace('\hline','')
    print(Latex_str)
    
    return(AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class)

def FineTuneModel(model,dataset,df,x_col,y_col,path_im,str_val,num_classes,epochs=20,Net='VGG'):
    """
    @param x_col : name of images
    @param y_col : classes
    @param path_im : path to images
    """
    df_train = df[df['set']=='train']
    df_val = df[df['set']==str_val]
    df_train[x_col] = df_train[x_col].apply(lambda x : x + '.jpg')
    df_val[x_col] = df_val[x_col].apply(lambda x : x + '.jpg')
    if len(df_val)==0:
        df = df[not(df['set']=='test')]
        df_train, df_val = train_test_split(df, test_size=0.33)
        
    if Net=='VGG' or Net=='VGGAdaIn':
        preprocessing_function = tf.keras.applications.vgg19.preprocess_input
    elif Net=='ResNet50':
        preprocessing_function = tf.keras.applications.resnet50.preprocess_input
    else:
        print(Net,'is unknwon')
        raise(NotImplementedError)
        
    datagen= tf.keras.preprocessing.image.ImageDataGenerator()
    train_generator=datagen.flow_from_dataframe(dataframe=df_train, directory=path_im,\
                                                x_col=x_col,y_col=y_col,\
                                                class_mode="other", \
                                                target_size=(224,224), batch_size=32,\
                                                shuffle=True,\
                                                preprocessing_function=preprocessing_function)
    valid_generator=datagen.flow_from_dataframe(dataframe=df_val, directory=path_im,\
                                                x_col=x_col,y_col=y_col,\
                                                class_mode="other", \
                                                target_size=(224,224), batch_size=32,\
                                                preprocessing_function=preprocessing_function)
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs)
    return(model)
    
def predictionFT_net(model,df_test,x_col,y_col,path_im,Net='VGG'):
    df_test[x_col] = df_test[x_col].apply(lambda x : x + '.jpg')
    datagen= tf.keras.preprocessing.image.ImageDataGenerator()
    
    if Net=='VGG' or Net=='VGGAdaIn':
        preprocessing_function = tf.keras.applications.vgg19.preprocess_input
    elif Net=='ResNet50':
        preprocessing_function = tf.keras.applications.resnet50.preprocess_input
    else:
        print(Net,'is unknwon')
        raise(NotImplementedError)
    test_generator=datagen.flow_from_dataframe(dataframe=df_test, directory=path_im,\
                                                x_col=x_col,\
                                                class_mode=None,shuffle=False,\
                                                target_size=(224,224), batch_size=1,
                                                preprocessing_function=preprocessing_function)
    predictions = model.predict_generator(test_generator)
    return(predictions)
    
def evaluationScoreDict(y_gt,dico_pred,verbose=False,k = 20,seuil=0.5):
    """
    @param k for precision at rank k
    @param the seuil can change a lot of things on the F1 score
    """
    num_samples,num_classes = y_gt.shape
    AP_per_class = []
    P_per_class = []
    R_per_class = []
    F1_per_class = []
    P20_per_class = []
    for c in range(num_classes):
        y_gt_c = y_gt[:,c]
        [y_predict_confidence_score,y_predict_test] = dico_pred[c]
        AP = average_precision_score(y_gt_c,y_predict_confidence_score,average=None)
        if verbose: print("Average Precision on all the data for classe",c," = ",AP)  
        AP_per_class += [AP] 
        test_precision = precision_score(y_gt_c,y_predict_test)
        test_recall = recall_score(y_gt_c,y_predict_test)
        R_per_class += [test_recall]
        P_per_class += [test_precision]
        F1 = f1_score(y_gt_c,y_predict_test)
        F1_per_class +=[F1]
        precision_at_k = ranking_precision_score(np.array(y_gt_c), y_predict_confidence_score,k)
        P20_per_class += [precision_at_k]
        if verbose: print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {2:.2f}, precision a rank k=20  = {3:.2f}.".format(test_precision,test_recall,F1,precision_at_k))
    return(AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class)
    
def evaluationScore(y_gt,y_pred,verbose=False,k = 20,seuil=0.5):
    """
    y_gt must be between 0 or 1
    y_predmust be between 0 and 1
    @param k for precision at rank k
    @param the seuil can change a lot of things
    """
    num_samples,num_classes = y_gt.shape
    AP_per_class = []
    P_per_class = []
    F1_per_class = []
    R_per_class = []
    P20_per_class = []
    for c in range(num_classes):
        y_gt_c = y_gt[:,c]
        y_predict_confidence_score = y_pred[:,c]
        y_predict_test = (y_predict_confidence_score>seuil).astype(int)
        AP = average_precision_score(y_gt_c,y_predict_confidence_score,average=None)
        if verbose: print("Average Precision on all the data for classe",c," = ",AP)  
        AP_per_class += [AP] 
        test_precision = precision_score(y_gt_c,y_predict_test)
        test_recall = recall_score(y_gt_c,y_predict_test)
        R_per_class += [test_recall]
        P_per_class += [test_precision]
        F1 = f1_score(y_gt_c,y_predict_test)
        F1_per_class +=[F1]
        precision_at_k = ranking_precision_score(np.array(y_gt_c), y_predict_confidence_score,k)
        P20_per_class += [precision_at_k]
        if verbose: print("Test on all the data precision = {0:.2f}, recall = {1:.2f}, F1 = {2:.2f}, precision a rank k=20  = {3:.2f}.".format(test_precision,test_recall,F1,precision_at_k))
    return(AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class)

def RunUnfreezeLayerPerformanceVGG():
    """
    The goal is to unfreeze only some part of the network
    """
    list_freezingType = ['FromTop','FromBottom','Alter']
    
    transformOnFinalLayer_tab = ['GlobalMaxPooling2D','GlobalAveragePooling2D','']
    for optimizer,opt_option in zip(['adam','SGD'],[[0.01],[0.1,0.01]]):
        for transformOnFinalLayer in transformOnFinalLayer_tab:
            for freezingType in list_freezingType:
                for pretrainingModif in range(1,17):
                    metrics = learn_and_eval(target_dataset='Paintings',constrNet='VGG',\
                                             kind_method='FT',epochs=20,transformOnFinalLayer=transformOnFinalLayer,\
                                             pretrainingModif=pretrainingModif,freezingType=freezingType,
                                             optimizer=optimizer,opt_option=opt_option)
        
                    AP_per_class,P_per_class,R_per_class,P20_per_class,F1_per_class = metrics
    
def RunAllEvaluation(dataset='Paintings',forLatex=False):
    target_dataset='Paintings'
    source_dataset = 'ImageNet'
    ## Run the baseline
    
    transformOnFinalLayer_tab = ['GlobalMaxPooling2D','GlobalAveragePooling2D']
    for normalisation in [False]:
        final_clf_list = ['LinearSVC'] # LinearSVC but also MLP
        features_list = ['fc2','fc1','flatten'] # We want to do fc2, fc1, max spatial and concat max and min spatial
         # We want to do fc2, fc1, max spatial and concat max and min spatial
        # Normalisation and not normalise
        kind_method = 'TL'
        style_layers = []
        
        # Baseline with just VGG
        constrNet = 'VGG'
        for final_clf in final_clf_list:
            for features in features_list:
                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers,gridSearch=False,
                           normalisation=normalisation,transformOnFinalLayer='')
            
            # Pooling on last conv block
            for transformOnFinalLayer in transformOnFinalLayer_tab:
                features = 'block5_pool'
                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                           constrNet,kind_method,style_layers,gridSearch=False,
                           normalisation=normalisation,transformOnFinalLayer=transformOnFinalLayer,\
                           forLatex=forLatex)
         
        # With VGGInNorm
        style_layers_tab_forOther = [['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
                         ['block1_conv1','block2_conv1'],['block1_conv1']]
        style_layers_tab_foVGGBaseNormCoherentr = [['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1'],
                         ['block1_conv1','block2_conv1']]
        
        features_list = ['fc2','fc1','flatten']
        features_list = ['fc2','fc1']
        net_tab = ['VGGInNorm','VGGInNormAdapt','VGGBaseNorm','VGGBaseNormCoherent']
        number_im_considered_tab = [1000]
        for getBeforeReLU in [True,False]:
            for constrNet in net_tab:
                if constrNet=='VGGBaseNormCoherent':
                    style_layers_tab = style_layers_tab_foVGGBaseNormCoherentr
                else:
                    style_layers_tab = style_layers_tab_forOther
                for final_clf in final_clf_list:
                    for style_layers in style_layers_tab:
                        for features in features_list:
                            for number_im_considered in number_im_considered_tab:
                                if not(forLatex): print('=== getBeforeReLU',getBeforeReLU,'constrNet',constrNet,'final_clf',final_clf,'features',features,'number_im_considered',number_im_considered,'style_layers',style_layers)
                                learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                                       constrNet,kind_method,style_layers,gridSearch=False,
                                       number_im_considered=number_im_considered,\
                                       normalisation=normalisation,getBeforeReLU=getBeforeReLU,\
                                       forLatex=forLatex)
                       
                        number_im_considered = 1000
                        # Pooling on last conv block
                        for transformOnFinalLayer in transformOnFinalLayer_tab:
                            if not(forLatex):  print('=== getBeforeReLU',getBeforeReLU,'constrNet',constrNet,'final_clf',final_clf,'features',features,'number_im_considered',number_im_considered,'style_layers',style_layers,'transformOnFinalLayer',transformOnFinalLayer)
                            features = 'block5_pool'
                            learn_and_eval(target_dataset,source_dataset,final_clf,features,\
                                       constrNet,kind_method,style_layers,gridSearch=False,
                                       number_im_considered=number_im_considered,
                                       normalisation=normalisation,getBeforeReLU=getBeforeReLU,\
                                       transformOnFinalLayer=transformOnFinalLayer,\
                                       forLatex=forLatex)
                    
if __name__ == '__main__': 
    # Ce que l'on 
    RunUnfreezeLayerPerformanceVGG()
    #RunAllEvaluation()
#    learn_and_eval(target_dataset='Paintings',constrNet='VGG',kind_method='FT',weights=None,epochs=20,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,pretrainingModif=False,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='ResNet50',kind_method='FT',epochs=20,pretrainingModif=False,transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)
###    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',kind_method='FT',\
##                   epochs=20,transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True)
##    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',kind_method='FT',\
#                   epochs=20,transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)
#    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',weights=None,\
#                   kind_method='FT',getBeforeReLU=True,epochs=20,\
#                   transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True,\
#                   style_layers= ['block1_conv1','block1_conv2','block2_conv1','block2_conv2',
#                                  'block3_conv1','block3_conv2','block3_conv3','block3_conv4',
#                                  'block4_conv1','block4_conv2','block4_conv3','block4_conv4', 
#                                  'block5_conv1','block5_conv2','block5_conv3','block5_conv4'])
#    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',weights='imagenet',\
#                   kind_method='FT',getBeforeReLU=True,epochs=20,\
#                   transformOnFinalLayer='GlobalMaxPooling2D',forLatex=True,\
#                   style_layers= ['block1_conv1','block1_conv2','block2_conv1','block2_conv2',
#                                  'block3_conv1','block3_conv2','block3_conv3','block3_conv4',
#                                  'block4_conv1','block4_conv2','block4_conv3','block4_conv4', 
#                                  'block5_conv1','block5_conv2','block5_conv3','block5_conv4'])
#    learn_and_eval(target_dataset='Paintings',constrNet='VGGAdaIn',\
#                   kind_method='FT',getBeforeReLU=True,epochs=20,transformOnFinalLayer='GlobalAveragePooling2D',forLatex=True)
