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
from collections import Counter
import matplotlib.pyplot as plt     
import pathlib
   
# Import libraries
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from urllib.error import HTTPError

from PIL import Image

path_data = '/media/gonthier/HDD2/data/Ukiyoe/'
path_data = '/media/gonthier/HDD2/data/Ukiyoe2/'

def downloadAndSave(original_url,image_id,folder,path_data = ''):
    #print(original_url)
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'Mozilla/5.0')
    local_filename, headers = opener.retrieve(original_url, 'Test.pdf')
    #local_filename, headers = urllib.request.urlretrieve(original_url)
    dst = os.path.join(path_data,folder,str(image_id)+'.jpg') 
    if os.path.isfile(dst): # If the file exist we will change its name
        i = 0
        while os.path.isfile(dst):
            dst = os.path.join(path_data,folder,str(image_id)+'_'+str(i)+'.jpg')
            i += 1
    copyfile(local_filename, dst) # We copy the file to a new folder


def count_number_of_images_WithsizeSuperiorTo(size=1024):
    """
    Count the number of images with a size superior to size**2
    """
    number_of_pixels = size**2
    databasetxt = path_data+'Ukiyoe_full_dataset.csv'
    path_im = path_data + '/' +'im'
    df = pd.read_csv(databasetxt,sep=",")
    number_total_img = len(df)
    print('Number of total images :',number_total_img)
    number_img_bigger = 0
    number_artwork_without_image = 0
    for row in df.iterrows():
        name_img = row['item_name']
        print(name_img)
        img_path = path_im + name_img + '.jpg'
        if os.path.isfile(img_path):
            im = Image.open(img_path)
            w, h = im.size
            h_int = int(h)
            w_int = int(w)
            if h_int*w_int >= number_of_pixels:
                number_img_bigger += 1 
        else: 
            number_artwork_without_image += 1 
    print('We have ',number_img_bigger,'images bigger than',size,'**2')
    print('We have ',number_artwork_without_image,'artworks without an image.')
        

def get_all_images(downloadImage=True):
    
    print('Beginning get_all_images function')
    
    df = pd.DataFrame(columns=['item_name','url','artist','title','date'])
    number_elt_per_page = 100
    
    url_with_key = "https://ukiyo-e.org/search?q="
    url_base = "https://ukiyo-e.org/search?"
    url_page = "start="
    per_page = "&per_page="+str(number_elt_per_page)
    
    list_im = []
    number_of_atworks_seen = 0
    json_counter = 0
    first_hit_url = url_with_key +per_page # Set the URL you want to webscrape from
    # Connect to the URL
    response = requests.get(first_hit_url)
    # Parse HTML and save to BeautifulSoup object¶
    soup = BeautifulSoup(response.text, "html.parser")
    # To download the whole data set, let's do a for loop through all a tags
    line_count = 1 #variable to track what line you are on
    
    
    for one_span_tag in soup.findAll('span'):
        try:
            classe = one_span_tag['class'] # class of the link
        except KeyError as e:
            classe = ['']
        if classe[0] == 'total':  
            strings = one_span_tag.contents
            #print(strings)
            numberPrints = strings[0].contents[0]
            #print(numberPrints,type(numberPrints))
            numberPrints_int = int(numberPrints.replace(',',''))
            #print(numberPrints_int,type(numberPrints_int))
            break

    nb_prints = 0
    nb_pages = 0
    
    if downloadImage:
        forbidden_images = open(path_data+'Forbidden_images.txt', 'w')
    
    while nb_prints < numberPrints_int:
        if not(nb_prints==0):
            nb_pages += 1 
            hit_url = url_base + url_page + str(nb_pages*number_elt_per_page)
            response = requests.get(hit_url)
            soup = BeautifulSoup(response.text, "html.parser")
        for one_a_tag in soup.findAll('a'):  #'div' tags are for links
            #print(one_a_tag)
            link = one_a_tag['href'] # Link to other pages
            try:
                classe = one_a_tag['class'] # class of the link
            except KeyError as e:
                classe = ['']
            #print(classe)
            
            if nb_prints % 1000 ==0:
                print(df)
                df.to_csv(path_data+'Ukiyoe_full_dataset.csv',sep=',', encoding="utf-8")
                
            if classe[0] == 'img': 
                MainImageFounded = False
                MainImageDownloaded = False
                
                # One new print
                main_url_img = ''
                artist = ''
                date = ''
                title= ''
                #print('enter image')
                #print('link',link)
                response_artwork = requests.get(link)
                #print(response_artwork)
                soup_artwork = BeautifulSoup(response_artwork.text, "html.parser")
                for one_a_tag_artwork in soup_artwork.findAll('a'):
                   url_img = one_a_tag_artwork['href'] 
                   if 'images' in url_img:
                       MainImageFounded = True
                       #print(url_img)
                       main_url_img = url_img
                for one_p_tag_artwork in soup_artwork.findAll('p'):
                    #print(one_p_tag_artwork)
                    try:
                        classe = one_p_tag_artwork['class'] # class of the link
                    except KeyError as e:
                        classe = ['']
                    #print(classe)
                    soup_one_p_tag_artwork = BeautifulSoup(one_p_tag_artwork.text, "html.parser")
                    #print('soup_one_p_tag_artwork',soup_one_p_tag_artwork.contents,type(soup_one_p_tag_artwork.contents))
                    if classe[0]=="row":
                        if classe[-1]=="artist":
                            #for op in soup_one_p_tag_artwork.findFirst('a'):
                            artist_raw = soup_one_p_tag_artwork.contents
                            #print(artist_raw)
                            artist= artist_raw[0].split(':')[-1]
                            artist = artist.replace('\t','')
                            artist = artist.replace('\n','')
                            #print('artist',artist)
                        if classe[-1]=="title":
                            title_raw = soup_one_p_tag_artwork.contents
                            title= title_raw[0].split(':')[-1]
                            title = title.replace('\t','')
                            title = title.replace('\n','')
                            #print('title',title)
                        if classe[-1]=="date":
                            date_raw = soup_one_p_tag_artwork.contents
                            date= date_raw[0].split(':')[-1]
                            date = date.replace('\t','')
                            date = date.replace('\n','')
                            #print('date',date)
                if MainImageFounded:
                    item_name = main_url_img.split('/')[-1]
                    item_name_tab = item_name.split('.')[0:-1]
                    item_name = '.'.join(item_name_tab)
                    df.loc[nb_prints] = [item_name,main_url_img,artist,title,date]
                    try:
                        if downloadImage:
                            downloadAndSave(main_url_img,item_name,'im',path_data)
                    except HTTPError as e:
                        print(e,'for',main_url_img)
                        forbidden_images.write(main_url_img)
                        forbidden_images.write('\n')
                        forbidden_images.flush()
                    nb_prints += 1
                    MainImageDownloaded = True
                if MainImageDownloaded:
                    continue
    if downloadImage:
        forbidden_images.close()     
    print(df)
    df.to_csv(path_data+'Ukiyoe_full_dataset.csv',sep=',', encoding="utf-8")
    df.to_csv(path_data+'Ukiyoe_full_dataset_copie.csv',sep=',', encoding="utf-8")
   

if __name__ == '__main__':                          
    get_all_images()
