#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:38:41 2017

@author: gonthier
"""

import os, os.path
import cv2
import numpy as np
import datetime
import scipy.io
from PIL import Image
import pathlib
import signal

# Register an handler for the timeout
def handler(signum, frame):
    print("Forever is over!")
    raise Exception("end of time")
     
    

def do_mkdir(path):
	if not(os.path.isdir(path)):
		os.mkdir(path)
	return(0)

def resizeRijkmuseum(bigger_size_tab=[229],data='',cv2bool = True):
    signal.signal(signal.SIGALRM, handler)
    # Define a timeout for my function
    signal.alarm(1200)
    print(data,str(bigger_size_tab))
    origin_path = '/media/gonthier/HDD/dataRaw/'
    folder = 'Rijksmuseum'+data+'/'
    read_path= origin_path + folder
    target_path = '/media/gonthier/HDD/data/'
    write_data = target_path + folder
    for bigger_size in bigger_size_tab:
        write_data_tmp = write_data + str(bigger_size) + '/'
        pathlib.Path(write_data_tmp).mkdir(parents=True, exist_ok=True) 
        #do_mkdir(write_data_tmp)
    i = 0
    valid_images = [".jpg",".png",".tga",".jpeg",'.tiff','.tif']
    list_img = os.listdir(read_path)
    print("Number of images :",len(list_img)-1)
    list_pb = []
    itera = 10000
    #itera = 1
    for f in list_img:
        if (i%itera==0):
            print(i,f)
        ext = os.path.splitext(f)[1]
        
        if ext.lower() not in valid_images:
            if not(f=='paitings_wikidata.csv'):
                print('extension problem',f)
                continue
            else:
                continue
        elif ext.lower()==".gif":
            to_open = read_path + f
            new_name = os.path.splitext(f)[0] +'.jpg'
            print('Gif !!',new_name)
            test = read_path + new_name
            if not(os.path.exists(test)):
                Image.open(to_open).convert('RGB').save(test)
            os.remove(to_open)
            f = new_name
            print("Gif Images, have been modified")
            input('wait !!!')
                
        to_open = read_path + f
        try:
            already_created = True
            for bigger_size in bigger_size_tab:
                write_data_tmp = write_data + str(bigger_size) + '/'
                name = write_data_tmp + os.path.splitext(f)[0] + '.jpg'
                already_created = already_created and os.path.exists(name)
            if not(already_created) and not(f in list_pb):
                now = datetime.datetime.now()
#                print(i,f, "at", now.year, now.month, now.day, now.hour, now.minute, now.second)
                if(cv2bool == True):
                    im = cv2.imread(to_open)
                else:
                    im = np.array(Image.open(to_open))
                    im =  im[:,:,::-1] 
                    #print("open")
                height, width = im.shape[:2]
                if not(height*width*3 > 2**31-1):
                    # To shift from BGR to RGB = > this implementation of VGG take RGB image
                    for bigger_size in bigger_size_tab:
                        write_data_tmp = write_data + str(bigger_size) + '/'
                        name = write_data_tmp + os.path.splitext(f)[0] + '.jpg'
                        if not(os.path.exists(name)):
                            if(height < width):
                                dim = (bigger_size, int(width *bigger_size*1.0 / height))
                            else:
                                dim = (int(height *bigger_size*1.0 / width),bigger_size)
                            tmp = (dim[1],dim[0])
                            dim = tmp
                            resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
                            cv2.imwrite(name,resized)
                            #print(f)
                else:
                    print('Image too big ',i,to_open,height, width)
                    continue
            i += 1
        except AttributeError:
            print('AttributeError for ',to_open)
            continue
        except cv2.error as e:
            print('Error at image ',i,to_open,height, width )
            continue
        except Exception as exc: 
            print(exc,f)
            continue
#        except:
#            print('Error at image ',i,to_open,height, width )
#            return(0)
    print("Number of images : ",i)
    
    return(0)

def resize(bigger_size_tab=[229],data='Prints',cv2bool = True):
    signal.signal(signal.SIGALRM, handler)
    # Define a timeout for my function
    signal.alarm(1200)
    print(data,str(bigger_size_tab))
    origin_path = '/media/gonthier/HDD/dataRaw/'
    folder = 'Wikidata_'+data+'/'
    read_path= origin_path + folder
    target_path = '/media/gonthier/HDD/data/'
    write_data = target_path + folder
    for bigger_size in bigger_size_tab:
        write_data_tmp = write_data + str(bigger_size) + '/'
        do_mkdir(write_data_tmp)
    i = 0
    valid_images = [".jpg",".png",".tga",".jpeg",'.tiff','.tif']
    list_img = os.listdir(read_path)
    print("Number of images :",len(list_img)-1)
    list_pb = ['Fiskere ved stranden en stille sommeraften.jpg','Albert Gottschalk - Vinterdag i Lyngby.jpg','Gertrud Sabine Spengler by Eriksen.jpg','En mose ved Høsterkøb med tørvearbejdere.jpg','Eleonore Agnes Raben Gammel Estrup.jpg','Erik XIV i Fængsel.jpg']
    list_pb = []
    itera = 10000
    #itera = 1
    for f in list_img:
        if (i%itera==0):
            print(i,f)
        ext = os.path.splitext(f)[1]
        
        if ext.lower() not in valid_images:
            if not(f=='paitings_wikidata.csv'):
                print('extension problem',f)
                continue
            else:
                continue
        elif ext.lower()==".gif":
            to_open = read_path + f
            new_name = os.path.splitext(f)[0] +'.jpg'
            print('Gif !!',new_name)
            test = read_path + new_name
            if not(os.path.exists(test)):
                Image.open(to_open).convert('RGB').save(test)
            os.remove(to_open)
            f = new_name
            print("Gif Images, have been modified")
            input('wait !!!')
                
        to_open = read_path + f
        try:
            already_created = True
            for bigger_size in bigger_size_tab:
                write_data_tmp = write_data + str(bigger_size) + '/'
                name = write_data_tmp + os.path.splitext(f)[0] + '.jpg'
                already_created = already_created and os.path.exists(name)
            if not(already_created) and not(f in list_pb):
                now = datetime.datetime.now()
                print(i,f, "at", now.year, now.month, now.day, now.hour, now.minute, now.second)
                if(cv2bool == True):
                    im = cv2.imread(to_open)
                else:
                    im = np.array(Image.open(to_open))
                    im =  im[:,:,::-1] 
                    #print("open")
                height, width = im.shape[:2]
                if not(height*width*3 > 2**31-1):
                    # To shift from BGR to RGB = > this implementation of VGG take RGB image
                    for bigger_size in bigger_size_tab:
                        write_data_tmp = write_data + str(bigger_size) + '/'
                        name = write_data_tmp + os.path.splitext(f)[0] + '.jpg'
                        if not(os.path.exists(name)):
                            if(height < width):
                                dim = (bigger_size, int(width *bigger_size*1.0 / height))
                            else:
                                dim = (int(height *bigger_size*1.0 / width),bigger_size)
                            tmp = (dim[1],dim[0])
                            dim = tmp
                            resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
                            cv2.imwrite(name,resized)
                            #print(f)
                else:
                    print('Image too big ',i,to_open,height, width)
                    continue
            i += 1
        except AttributeError:
            print('AttributeError for ',to_open)
            continue
        except cv2.error as e:
            print('Error at image ',i,to_open,height, width )
            continue
        except Exception as exc: 
            print(exc,f)
            continue
#        except:
#            print('Error at image ',i,to_open,height, width )
#            return(0)
    print("Number of images : ",i)
    
    return(0)
    
if __name__ == '__main__':
    #bigger_size_tab = [224,256,299,340,600]
    #resize(bigger_size_tab=bigger_size_tab,data='Prints')
    bigger_size_tab = [224,229,256,299,340,600]
    #resize(bigger_size_tab=bigger_size_tab,data='Paintings',cv2bool =True) #70993 attendus
    resizeRijkmuseum(bigger_size_tab=bigger_size_tab,data='/im',cv2bool = True)
    # Images too big : 
    # 
    # 
    
    # Traiter :
    # Jürgen Ovens - Justice (or Prudence, Justice, and Peace) - Google Art Project.jpg
    # Giovanni Bellini - Saint Francis in the Desert - Google Art Project.jpg
    # BoschTheCrucifixionOfStJulia.jpg
    # Artemisia, by Rembrandt, from Prado in Google Earth.jpg
    # Hieronymus Bosch - Hermit Saints Triptych.jpg
    # Pierre-Denis Martin - View of the Château de Fontainebleau - Google Art Project.jpg
    # Bernat Martorell - Altarpiece of Saint Vincent - Google Art Project.jpg
    # The Three Graces, by Peter Paul Rubens, from Prado in Google Earth.jpg
    # Hans Holbein the Younger - The Ambassadors - Google Art Project.jpg
    
    # Image qui cause des problemes : 
    # Erik XIV i Fængsel.jpg
    # Eleonore Agnes Raben Gammel Estrup.jpg
    # En mose ved Høsterkøb med tørvearbejdere.jpg
    # Albert Gottschalk - Vinterdag i Lyngby.jpg
    # Fiskere ved stranden en stille sommeraften.jpg