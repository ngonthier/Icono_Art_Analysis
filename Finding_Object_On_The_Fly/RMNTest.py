# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:44:03 2019

The goal of this script is to read the RMN GP API and to get the images with 
the keywords

@author: gonthier
"""


import json
import urllib
from shutil import copyfile
import os
import pandas as pd
import pickle
import gc
import csv
import requests
import numpy as np
import glob
import shutil
        
def get_jsonold(url):
    response = urllib.request.urlopen(url)
    data = response.read()
    values = json.loads(data)
    return(values)
    
def get_json(url):
    response = requests.get(url)
    data = response.text
    values = json.loads(data)
    return(values)
    
path_data = 'Test'
path_data = '/media/gonthier/HDD/data/RMN'

def downloadAndSave(image,folder,path_data = ''):
    urls = image['urls']
    original_url = urls['original']
    image_id = image['id']
    local_filename, headers = urllib.request.urlretrieve(original_url)
    dst = os.path.join(path_data,folder,str(image_id)+'.jpg') 
    if os.path.isfile(dst): # If the file exist we will change its name
        i = 0
        while os.path.isfile(dst):
            dst = os.path.join(path_data,folder,str(image_id)+'_'+str(i)+'.jpg')
            i += 1
    copyfile(local_filename, dst) # We copy the file to a new folder



#url = "https://api.art.rmngp.fr/v1/works?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd&facets[techniques]=huile+sur+toile"
#url = "https://api.art.rmngp.fr/v1/works?page=2?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd&facets[techniques]=huile+sur+toile"
#url = "https://api.art.rmngp.fr:443/v1/works?page=2&&&&&&&&&&&&&&&&&?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd"
#url_with_key = "https://api.art.rmngp.fr/v1/works?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd&page=3&facets[techniques]=huile+sur+toile"

def generateJSON_and_urlsFiles():

    number_elt_per_page = 250
    
    url_with_key = "https://api.art.rmngp.fr/v1/works?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd"
    technique_huile_sur_toile = "&facets[techniques]=huile+sur+toile"
    technique_huile_sur_bois = "&facets[techniques]=huile+sur+bois"
    url_page = "&page="
    per_page = "&per_page="+str(number_elt_per_page)
    period = "&facets[periods]="
    siecle = "e+siècle"
    #siecle = "e+si\xc3\xa8cle"
    list_siecle = [13,14,15,16,17,18,19,20,21]
    list_technique = [technique_huile_sur_toile,technique_huile_sur_bois]
    
    #response = urllib.request.urlopen(first_hit_url)
    #data = response.read()
    #values = json.loads(data)

    SaveDetails = False
    
    number_of_atworks_seen = 0 #+(p-1)*number_elt_per_page
    list_im_keywords = []
    # Pour une raison obscure 1057046 retourne un detail...
    
    df = pd.DataFrame(columns=['oeuvre','name','item','kewords'])
    name_csv_huile = os.path.join(path_data,'Huiles.csv')
    
    list_unfound = []
    list_image_to_download = []
    urlsFile = os.path.join(path_data,"urls_im.csv")
    
    json_counter = 0
    
    for technique in list_technique:
        for time in list_siecle:
            print(technique,time)
            first_hit_url = url_with_key + technique+period+str(time)+siecle +per_page
            #print(first_hit_url)
            values = get_json(first_hit_url)
            total_number_of_item = values['hits']['total']
            p = 1
            number_of_atworks_seen_local = 0
            while number_of_atworks_seen_local < total_number_of_item:
                print('Page :',p,'number_of_atworks_seen',number_of_atworks_seen)
                if not(p==1):
                    url = url_with_key + technique+period+str(time)+siecle + url_page+ str(p)+per_page 
                    #print(url)
                    values = get_json(url)
                    #print(values)
                    gc.collect()
                json_name = os.path.join(path_data,'JSON','HST'+str(json_counter)+'.json')
                json_counter += 1
                with open(json_name, 'w') as json_file:  
                    json.dump(values, json_file)
                hits = values['hits']['hits']  
            
                for hit in hits:
                    number_of_atworks_seen += 1
                    number_of_atworks_seen_local += 1
                    id_oeuvre = hit['_id']
                    source = hit['_source']
                    slug = source['slug']
                    
                    techniques = source['techniques']
                    name_tech = techniques[0]['name']['fr']
                    
                    try:
                        main_image = source['image']
                        main_image_id = main_image['id']
                    except KeyError:  
                        main_image_id = None
                        pass
                    images = source['images']
                    print(number_of_atworks_seen,' : ',id_oeuvre,main_image_id,slug)
                    list_keywords = []
                    for image in images:
                        image_id = image['id']
                        #print('image :',image_id)
                        if main_image_id is None: # Si il n y a pas d images principales associées on prend la premiere
                            main_image_id = image_id
                        # Boolean, true if this is the main work image
                        if image_id==main_image_id:
                            folder = 'IM'
                            list_image_to_download += [[image['urls']['original']]]
            #                try:
            #                    downloadAndSave(image,folder,path_data = path_data)
            #                    gc.collect()
            #                except urllib.error.HTTPError:
            #                    list_unfound +=[[id_oeuvre,slug,main_image_id,images]]
                            try:
                                keywords = image['keywords']
                                for k in keywords:
                                    keyword = k['name']['fr']
                                    list_keywords += [keyword]                    
            #                    couple = [image_id,list_keywords] # Attention on recupere juste les details associes a la grande image et pas aux imagettes
            #                    list_im_keywords += [couple]
                            except KeyError:
                                pass
                        else:
                            try:
                                keywords = image['keywords']
                                for k in keywords:
                                    keyword = k['name']['fr']
                                    list_keywords += [keyword]                    
            #                    couple = [image_id,list_keywords] # Attention on recupere juste les details associes a la grande image et pas aux imagettes
            #                    list_im_keywords += [couple]
                            except KeyError:
                                pass
                            if SaveDetails:
                                folder = 'IMdetails'
                                downloadAndSave(image,folder,path_data = path_data)
                    df.loc[number_of_atworks_seen] = [id_oeuvre,slug,main_image_id,list_keywords]
                    gc.collect()
                    
                p += 1
                df.to_csv(name_csv_huile,index=False)
                with open(urlsFile,'w') as resultFile:
                    wr = csv.writer(resultFile,delimiter=',',\
                                        quotechar='\n', quoting=csv.QUOTE_MINIMAL)
                    wr.writerows(list_image_to_download)
                gc.collect()
    #    pickle_unfounded = os.path.join(path_data,'unfounded.pkl')
    #    if os.path.exists(pickle_unfounded):
    #        # "with" statements are very handy for opening files. 
    #        with open(pickle_unfounded,'rb') as rfp: 
    #            old = pickle.load(rfp)
    #        list_unfound.append(old)
    #    if len(list_unfound) >0:
    #        with open(pickle_unfounded, 'wb') as pkl_file:   
    #            pickle.dump(list_unfound,pkl_file)
    
    with open(urlsFile,'w') as resultFile:
        wr = csv.writer(resultFile,delimiter=',',\
                                quotechar='\n', quoting=csv.QUOTE_MINIMAL)
        wr.writerows(list_image_to_download)
        
    df.to_csv(name_csv_huile,index=False)

#gc.collect()
#pickle_unfounded = os.path.join(path_data,'unfounded.pkl')
#if os.path.exists(pickle_unfounded):
#    # "with" statements are very handy for opening files. 
#    with open(pickle_unfounded,'rb') as rfp: 
#        old = pickle.load(rfp)
#    list_unfound.append(old)
#if len(list_unfound) >0:
#    with open(pickle_unfounded, 'wb') as pkl_file:   
#        pickle.dump(list_unfound,pkl_file)
    
# Select the images
def SuppressDoublonIm():
    
    name_csv_huile = os.path.join(path_data,'Huiles.csv')
    df = pd.read_csv(name_csv_huile)
    dfvalues = df.values
    oeuvres = dfvalues[0,:]
    dfvalues.shape
    oeuvres = dfvalues[:,0]
    oeuvres.shape
    unique_oeuvres = np.unique(oeuvres)
    import numpy as np
    unique_oeuvres = np.unique(oeuvres)
    print('Len Unique Oeuvre :',len(unique_oeuvres))
    names = dfvalues[:,1]
    names.shape
    uniques_names = np.unique(names)
    print('Len Unique Names :',len(uniques_names))
    item = dfvalues[:,2]
    uniques_item = np.unique(item)
    print('Len Unique Images item :',len(uniques_item))
    keywords = dfvalues[:,3]
    
    im_raw_path = os.path.join(path_data,'IM','*')
    list_im_raw = glob.glob(im_raw_path)
    new_list = []
    for elt in list_im_raw:
        splitname = elt.split('?')[0]
        new_im = splitname.replace('IM','JPEGImages') + '.jpg'
        shutil.copyfile(elt,new_im) # Source Destination
        new_list += [splitname]
        
    unique_new_list = np.unique(new_list)
    print('Len Unique Downloaded images : ',len(unique_new_list))

    
    