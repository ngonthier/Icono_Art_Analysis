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

def get_json(url):
    response = urllib.request.urlopen(url)
    data = response.read()
    values = json.loads(data)
    return(values)
path_data = 'Test'
def downloadAndSave(image,folder,path_data = ''):
    urls = image['urls']
    original_url = urls['original']
    local_filename, headers = urllib.request.urlretrieve(original_url)
    dst = os.path.join(path_data,folder,str(image_id)+'.jpg') 
    copyfile(local_filename, dst) # We copy the file to a new folder



#url = "https://api.art.rmngp.fr/v1/works?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd&facets[techniques]=huile+sur+toile"
#url = "https://api.art.rmngp.fr/v1/works?page=2?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd&facets[techniques]=huile+sur+toile"
#url = "https://api.art.rmngp.fr:443/v1/works?page=2&&&&&&&&&&&&&&&&&?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd"
#url_with_key = "https://api.art.rmngp.fr/v1/works?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd&page=3&facets[techniques]=huile+sur+toile"
url_with_key = "https://api.art.rmngp.fr/v1/works?api_key=e511996f5894226e9fa1eb9593c650f0d49de7ba605f2cc60a928af49f30c0fd"
technique_huile_sur_toile = "&facets[techniques]=huile+sur+toile"
technique_huile_sur_bois = "&facets[techniques]=huile+sur+bois"
url_page = "&page="

first_hit_url = url_with_key + technique_huile_sur_toile
#response = urllib.request.urlopen(first_hit_url)
#data = response.read()
#values = json.loads(data)

values = get_json(first_hit_url)
total_number_of_item = values['hits']['total']
number_elt_per_page = 10
SaveDetails = False
p = 1
number_of_atworks_seen = 0
list_im_keywords = []
# Pour une raison obscure 1057046 retourne un detail...

while number_of_atworks_seen < total_number_of_item:
    print('Page :',p)
    if not(p==1):
        url = url_with_key +url_page+ str(p) + technique_huile_sur_toile
        values = get_json(url)
    json_name = os.path.join(path_data,'JSON','HST'+str(p)+'.json')
    with open(json_name, 'w') as json_file:  
        json.dump(values, json_file)
    hits = values['hits']['hits']  

    for hit in hits:
        number_of_atworks_seen += 1
        id_oeuvre = hit['_id']
        source = hit['_source']
        slug = source['slug']
        print(id_oeuvre,slug)
        techniques = source['techniques']
        name_tech = techniques[0]['name']['fr']
        
        main_image = source['image']
        main_image_id = main_image['id']
        images = source['images']
        for image in images:
            image_id = image['id']
            # Boolean, true if this is the main work image
            if image_id==main_image_id:
                folder = 'IM'
                downloadAndSave(image,folder,path_data = path_data)
                list_keywords = []
                try:
                    keywords = image['keywords']
                    for k in keywords:
                        keyword = k['name']['fr']
                        list_keywords += [keyword]
                    print(list_keywords)
                    couple = [image_id,list_keywords] # Attention on recupere juste les details associes a la grande image et pas aux imagettes
                    list_im_keywords += [couple]
                except KeyError:
                    pass
            elif SaveDetails:
                folder = 'IMdetails'
                downloadAndSave(image,folder,path_data = path_data)
            
    p += 1

