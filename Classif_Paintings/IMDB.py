#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:54:36 2019

@author: gonthier
"""

import pandas as pd
import os

def get_database(database,verbose=False):
    ext = '.txt'
    default_path_imdb = '/media/gonthier/HDD/data/'
    default_path_imdb2 = '/media/gonthier/HDD2/data/'
    if database=='Paintings':
        # Only for classification
        item_name = 'name_img'
        path_to_img = 'Painting_Dataset/'
        classes = ['aeroplane','bird','boat','chair','cow','diningtable','dog','horse','sheep','train']
    elif database=='RASTA':
        path_to_img = 'RASTA_LAMSADE/wikipaintings_full'
        ext = '.csv'
        item_name = 'name_img'
        default_path_imdb = default_path_imdb2
        classes=['Abstract_Art','Abstract_Expressionism','Art_Informel','Art_Nouveau_(Modern)','Baroque','Color_Field_Painting','Cubism','Early_Renaissance','Expressionism','High_Renaissance','Impressionism','Magic_Realism','Mannerism_(Late_Renaissance)','Minimalism','Naïve_Art_(Primitivism)','Neoclassicism','Northern_Renaissance','Pop_Art','Post-Impressionism','Realism','Rococo','Romanticism','Surrealism','Symbolism','Ukiyo-e']
    elif database=='VOC12':
        item_name = 'name_img'
        path_to_img = 'VOCdevkit/VOC2012/JPEGImages/'
    elif database=='VOC2007':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'VOCdevkit/VOC2007/JPEGImages/'
        classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    elif database=='watercolor':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'cross-domain-detection/datasets/watercolor/JPEGImages/'
        classes =  ["bicycle", "bird","car", "cat", "dog", "person"]
    elif database=='PeopleArt':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'PeopleArt/JPEGImages/'
        classes =  ["person"]
    elif database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        ext = '.csv'
        item_name = 'item'
        path_to_img = 'Wikidata_Paintings/WikiTenLabels/JPEGImages/'
        classes =  ['angel', 'beard','capital','Child_Jesus', 'crucifixion_of_Jesus',
                    'Mary','nudity', 'ruins','Saint_Sebastien','turban']
    elif 'IconArt_v1' in database:
            ext='.csv'
            item_name='item'
            classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
            'Mary','nudity', 'ruins','Saint_Sebastien']
            path_to_img = 'Wikidata_Paintings/IconArt_v1/JPEGImages/'
    elif(database=='RMN'):
            ext='.csv'
            item_name='item'
            classes =  ['angel','Child_Jesus', 'crucifixion_of_Jesus',
            'Mary','nudity', 'ruins','Saint_Sebastien']
            path_to_img = 'RMN/JPEGImages/'
    elif database=='clipart':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'cross-domain-detection/datasets/clipart/JPEGImages/'
        classes =  ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    elif database=='comic':
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'cross-domain-detection/datasets/comic/JPEGImages/'
        classes =  ['bicycle','bird','car','cat','dog','person']
    elif database=='CASPApaintings':
        default_path_imdb = default_path_imdb2
        ext = '.csv'
        item_name = 'name_img'
        path_to_img = 'CASPApaintings/JPEGImages/'     
        classes = ["bear", "bird", "cat", "cow", "dog", "elephant", "horse", "sheep"]
    elif(database=='Wikidata_Paintings'):
        item_name = 'image'
        path_to_img = 'data/Wikidata_Paintings/600/'
        print(database,' is not implemented yet')
        raise NotImplementedError # TODO implementer cela !!! 
    elif(database=='Wikidata_Paintings_miniset_verif'):
        item_name = 'image'
        path_to_img = 'Wikidata_Paintings/600/'
        classes = ['Q235113_verif','Q345_verif','Q10791_verif','Q109607_verif','Q942467_verif']
    elif 'OIV5' in database:
        # Possible OIV5_small_3135 and  OIV5_small_30001
        ext = '.csv'
        item_name = 'item'
        path_to_img = 'OIV5/Images/'
        default_path_imdb = default_path_imdb2
        classes = ['/m/011k07', '/m/0120dh', '/m/01226z', '/m/012n7d', '/m/012w5l',
           '/m/0130jx', '/m/0138tl', '/m/013y1f', '/m/014j1m', '/m/014sv8',
           '/m/014y4n', '/m/0152hh', '/m/01599', '/m/015h_t', '/m/015p6',
           '/m/015qff', '/m/015wgc', '/m/015x4r', '/m/015x5n', '/m/0162_1',
           '/m/0167gd', '/m/016m2d', '/m/0174k2', '/m/0174n1', '/m/0175cv',
           '/m/0176mf', '/m/017ftj', '/m/018p4k', '/m/018xm', '/m/01940j',
           '/m/0199g', '/m/019dx1', '/m/019h78', '/m/019jd', '/m/019w40',
           '/m/01_5g', '/m/01b638', '/m/01b7fy', '/m/01b9xk', '/m/01bfm9',
           '/m/01bjv', '/m/01bl7v', '/m/01bms0', '/m/01bqk0', '/m/01btn',
           '/m/01c648', '/m/01cmb2', '/m/01d40f', '/m/01dws', '/m/01dwsz',
           '/m/01dwwc', '/m/01dxs', '/m/01dy8n', '/m/01f8m5', '/m/01f91_',
           '/m/01fb_0', '/m/01fdzj', '/m/01fh4r', '/m/01g317', '/m/01g3x7',
           '/m/01gkx_', '/m/01gllr', '/m/01gmv2', '/m/01h3n', '/m/01h44',
           '/m/01h8tj', '/m/01hrv5', '/m/01j3zr', '/m/01j51', '/m/01j5ks',
           '/m/01j61q', '/m/01jfm_', '/m/01jfsr', '/m/01k6s3', '/m/01kb5b',
           '/m/01knjb', '/m/01krhy', '/m/01lcw4', '/m/01llwg', '/m/01lrl',
           '/m/01lsmm', '/m/01lynh', '/m/01m2v', '/m/01m4t', '/m/01mqdt',
           '/m/01mzpv', '/m/01n4qj', '/m/01n5jq', '/m/01nq26', '/m/01pns0',
           '/m/01prls', '/m/01r546', '/m/01rkbr', '/m/01rzcn', '/m/01s105',
           '/m/01s55n', '/m/01tcjp', '/m/01vbnl', '/m/01x3jk', '/m/01x3z',
           '/m/01x_v', '/m/01xq0k1', '/m/01xqw', '/m/01xs3r', '/m/01xygc',
           '/m/01xyhv', '/m/01y9k5', '/m/01yrx', '/m/01yx86', '/m/01z1kdw',
           '/m/02068x', '/m/020jm', '/m/020kz', '/m/020lf', '/m/021mn',
           '/m/021sj1', '/m/0220r2', '/m/0242l', '/m/024g6', '/m/02522',
           '/m/025dyy', '/m/025nd', '/m/025rp__', '/m/026qbn5', '/m/026t6',
           '/m/0270h', '/m/0271t', '/m/027pcv', '/m/0283dt1', '/m/029b3',
           '/m/029bxz', '/m/029tx', '/m/02_n6y', '/m/02crq1', '/m/02ctlc',
           '/m/02cvgx', '/m/02d1br', '/m/02d9qx', '/m/02dgv', '/m/02dl1y',
           '/m/02f9f_', '/m/02fq_6', '/m/02g30s', '/m/02gzp', '/m/02h19r',
           '/m/02hj4', '/m/02jfl0', '/m/02jnhm', '/m/02jvh9', '/m/02jz0l',
           '/m/02l8p9', '/m/02lbcq', '/m/02p3w7d', '/m/02p5f1q', '/m/02pdsw',
           '/m/02pjr4', '/m/02pkr5', '/m/02pv19', '/m/02rdsp', '/m/02rgn06',
           '/m/02s195', '/m/02tsc9', '/m/02vqfm', '/m/02w3_ws', '/m/02w3r3',
           '/m/02wbtzl', '/m/02wv6h6', '/m/02wv84t', '/m/02x8cch',
           '/m/02x984l', '/m/02xwb', '/m/02y6n', '/m/02z51p', '/m/02zn6n',
           '/m/02zt3', '/m/02zvsm', '/m/030610', '/m/0306r', '/m/03120',
           '/m/0319l', '/m/031b6r', '/m/031n1', '/m/0323sq', '/m/032b3c',
           '/m/033cnk', '/m/033rq4', '/m/0342h', '/m/034c16', '/m/035r7c',
           '/m/0388q', '/m/039xj_', '/m/03__z0', '/m/03bbps', '/m/03bk1',
           '/m/03bt1vf', '/m/03c7gz', '/m/03d443', '/m/03dnzn', '/m/03fj2',
           '/m/03fp41', '/m/03fwl', '/m/03g8mr', '/m/03grzl', '/m/03hl4l9',
           '/m/03jbxj', '/m/03jm5', '/m/03k3r', '/m/03kt2w', '/m/03ldnb',
           '/m/03m3pdh', '/m/03m3vtv', '/m/03m5k', '/m/03nfch', '/m/03p3bw',
           '/m/03q5c7', '/m/03q5t', '/m/03q69', '/m/03qrc', '/m/03rszm',
           '/m/03s_tn', '/m/03ssj5', '/m/03tw93', '/m/03v5tg', '/m/03vt0',
           '/m/03xxp', '/m/03y6mg', '/m/040b_t', '/m/04169hn', '/m/0420v5',
           '/m/043nyj', '/m/0449p', '/m/044r5d', '/m/046dlr', '/m/047j0r',
           '/m/047v4b', '/m/04_sv', '/m/04bcr3', '/m/04brg2', '/m/04c0y',
           '/m/04ctx', '/m/04dr76w', '/m/04g2r', '/m/04gth', '/m/04h7h',
           '/m/04h8sr', '/m/04hgtk', '/m/04kkgm', '/m/04m6gz', '/m/04m9y',
           '/m/04p0qw', '/m/04rmv', '/m/04szw', '/m/04tn4x', '/m/04v6l4',
           '/m/04vv5k', '/m/04y4h8h', '/m/04ylt', '/m/04yqq2', '/m/04yx4',
           '/m/04zwwv', '/m/050gv4', '/m/050k8', '/m/052sf', '/m/05441v',
           '/m/054_l', '/m/054fyh', '/m/054xkw', '/m/057cc', '/m/057p5t',
           '/m/0584n8', '/m/058qzx', '/m/05_5p_0', '/m/05bm6', '/m/05ctyq',
           '/m/05gqfk', '/m/05kms', '/m/05kyg_', '/m/05n4y', '/m/05r5c',
           '/m/05r655', '/m/05vtc', '/m/05z55', '/m/05z6w', '/m/05zsy',
           '/m/061_f', '/m/061hd_', '/m/0633h', '/m/063rgb', '/m/0642b4',
           '/m/0663v', '/m/068zj', '/m/06_72j', '/m/06__v', '/m/06_fw',
           '/m/06bt6', '/m/06c54', '/m/06j2d', '/m/06k2mb', '/m/06m11',
           '/m/06mf6', '/m/06msq', '/m/06ncr', '/m/06nrc', '/m/06nwz',
           '/m/06pcq', '/m/06y5r', '/m/06z37_', '/m/07030', '/m/0703r8',
           '/m/071p9', '/m/071qp', '/m/073bxn', '/m/076bq', '/m/076lb9',
           '/m/078jl', '/m/078n6m', '/m/079cl', '/m/07bgp', '/m/07c52',
           '/m/07c6l', '/m/07clx', '/m/07cmd', '/m/07crc', '/m/07cx4',
           '/m/07dd4', '/m/07dm6', '/m/07fbm7', '/m/07gql', '/m/07j7r',
           '/m/07j87', '/m/07jdr', '/m/07kng9', '/m/07mhn', '/m/07qxg_',
           '/m/07r04', '/m/07v9_z', '/m/07xyvk', '/m/07y_7', '/m/07yv9',
           '/m/080hkjn', '/m/081qc', '/m/083kb', '/m/083wq', '/m/084rd',
           '/m/084zz', '/m/0898b', '/m/08hvt4', '/m/08pbxl', '/m/096mb',
           '/m/09728', '/m/099ssp', '/m/09b5t', '/m/09csl', '/m/09ct_',
           '/m/09d5_', '/m/09ddx', '/m/09dzg', '/m/09f_2', '/m/09g1w',
           '/m/09gtd', '/m/09j5n', '/m/09k_b', '/m/09kmb', '/m/09kx5',
           '/m/09ld4', '/m/09qck', '/m/09rvcxw', '/m/09tvcd', '/m/0_cp5',
           '/m/0_k2', '/m/0b3fp9', '/m/0b_rs', '/m/0bh9flk', '/m/0bjyj5',
           '/m/0bt9lr', '/m/0bt_c3', '/m/0bwd_0j', '/m/0by6g', '/m/0c06p',
           '/m/0c29q', '/m/0c568', '/m/0c9ph5', '/m/0c_jw', '/m/0ccs93',
           '/m/0cd4d', '/m/0cdl1', '/m/0cdn1', '/m/0cffdh', '/m/0cgh4',
           '/m/0ch_cf', '/m/0cjq5', '/m/0cjs7', '/m/0cmf2', '/m/0cmx8',
           '/m/0cn6p', '/m/0cnyhnx', '/m/0crjs', '/m/0cvnqh', '/m/0cxn2',
           '/m/0cydv', '/m/0cyf8', '/m/0cyfs', '/m/0cyhj_', '/m/0czz2',
           '/m/0d20w4', '/m/0d4v4', '/m/0d5gx', '/m/0d8zb', '/m/0d_2m',
           '/m/0dbvp', '/m/0dbzx', '/m/0dftk', '/m/0dj6p', '/m/0djtd',
           '/m/0dkzw', '/m/0dq75', '/m/0dt3t', '/m/0dtln', '/m/0dv5r',
           '/m/0dv77', '/m/0dv9c', '/m/0dzct', '/m/0dzf4', '/m/0f4s2w',
           '/m/0f6wt', '/m/0f9_l', '/m/0fbdv', '/m/0fbw6', '/m/0fj52s',
           '/m/0fldg', '/m/0fly7', '/m/0fm3zh', '/m/0fp6w', '/m/0fqfqc',
           '/m/0fqt361', '/m/0frqm', '/m/0fszt', '/m/0ft9s', '/m/0ftb8',
           '/m/0fx9l', '/m/0fz0h', '/m/0gd2v', '/m/0gd36', '/m/0gj37',
           '/m/0gjbg72', '/m/0gjkl', '/m/0gm28', '/m/0grw1', '/m/0gv1x',
           '/m/0gxl3', '/m/0h23m', '/m/0h2r6', '/m/0h8l4fh', '/m/0h8lkj8',
           '/m/0h8mhzd', '/m/0h8my_4', '/m/0h8mzrc', '/m/0h8n27j',
           '/m/0h8n5zk', '/m/0h8n6f9', '/m/0h8n6ft', '/m/0h8ntjv',
           '/m/0h99cwc', '/m/0h9mv', '/m/0hdln', '/m/0hf58v5', '/m/0hg7b',
           '/m/0hkxq', '/m/0hnnb', '/m/0hqkz', '/m/0jbk', '/m/0jg57',
           '/m/0jly1', '/m/0jqgx', '/m/0jwn_', '/m/0jy4k', '/m/0jyfg',
           '/m/0k0pj', '/m/0k1tl', '/m/0k4j', '/m/0k5j', '/m/0k65p',
           '/m/0kmg4', '/m/0kpqd', '/m/0l14j_', '/m/0l515', '/m/0ll1f78',
           '/m/0llzx', '/m/0lt4_', '/m/0mkg', '/m/0mw_6', '/m/0n28_',
           '/m/0nl46', '/m/0nybt', '/m/0pcr', '/m/0pg52', '/m/0ph39',
           '/m/0qmmr', '/m/0wdt60w', '/m/0zvk5']
        # 38052 in total and 3135 for train
    elif database=='Ukiyoe_simple': # Simple and quick version of the Ukiyoe dataset
        # With 
        default_path_imdb = default_path_imdb2
        ext = '.csv'
        item_name = 'item_name'
        path_to_img = 'Ukiyoe/im/'     
        classes = ["1", "2", "3", "4", "5", "6"]
    else:
        print('This database don t exist :',database)
        raise NotImplementedError
    num_classes = len(classes)
    
    path_data = '/media/gonthier/HDD/output_exp/ClassifPaintings/'
    Not_on_NicolasPC = False
    if not(os.path.exists(path_data)): # Thats means you are not on the Nicolas Computer
        # Modification of the path used
        Not_on_NicolasPC = True
        if verbose: print('you are not on the Nicolas PC, so I think you have the data in the data folder')
        path_tmp = 'data/' 
        path_to_img = path_tmp + path_to_img
        path_data = path_tmp + 'ClassifPaintings/'
        if not(os.path.isdir(path_data)):
            path_data = path_tmp
            if not(path_data):
                print('The path the to data folder is not in this machine')
                raise(ValueError)
        if 'IconArt_v1' in database:
            path_data_csvfile = path_tmp+'Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
        elif database=='RMN':
            path_data_csvfile = path_tmp+'RMN/ImageSets/Main/'
        elif 'OIV5' in database:
            path_data_csvfile = path_tmp+'OIV5/'           
        else:
            path_data_csvfile = path_data
        default_path_imdb = path_tmp
    else:
        path_to_img = default_path_imdb + path_to_img
#        path_to_img = '/media/gonthier/HDD/data/' + path_to_img
#        dataImg_path = '/media/gonthier/HDD/data/'
        if 'IconArt_v1' in database:
            path_data_csvfile = '/media/gonthier/HDD/data/Wikidata_Paintings/IconArt_v1/ImageSets/Main/'
        elif database=='RMN':
            path_data_csvfile = '/media/gonthier/HDD/data/RMN/ImageSets/Main/'
        elif 'OIV5' in database:
            path_data_csvfile = '/media/gonthier/HDD2/data/OIV5/'
        elif database=='RASTA':
            path_data_csvfile = '/media/gonthier/HDD2/data/RASTA_LAMSADE/wikipaintings_full/'
        elif 'Ukiyoe' in database:
            path_data_csvfile = '/media/gonthier/HDD2/data/Ukiyoe/'
        else:
            path_data_csvfile = path_data
    
    databasetxt = path_data_csvfile + database + ext

    if database in ['WikiTenLabels','MiniTrain_WikiTenLabels','WikiLabels1000training']:
        dtypes = {0:str,'item':str,'angel':int,'beard':int,'capital':int, \
                      'Child_Jesus':int,'crucifixion_of_Jesus':int,'Mary':int,'nudity':int,'ruins':int,'Saint_Sebastien':int,\
                      'turban':int,'set':str,'Anno':int}
    elif 'IconArt_v1' in database or 'IconArt_v1'==database or database=='RMN':
        dtypes = {0:str,'item':str,'angel':int,\
                      'Child_Jesus':int,'crucifixion_of_Jesus':int,'Mary':int,'nudity':int,\
                      'ruins':int,'Saint_Sebastien':int,\
                      'set':str,'Anno':int}
    elif database=='VOC2007':
        dtypes = {0:str,'name_img':str,'aeroplane':int, 'bicycle':int, 'bird':int, 'boat':int,\
                  'bottle':int, 'bus':int, 'car':int, 'cat':int, 'chair':int,\
                  'cow':int, 'diningtable':int, 'dog':int, 'horse':int,\
                  'motorbike':int, 'person':int, 'pottedplant':int,\
                  'sheep':int, 'sofa':int, 'train':int, 'tvmonitor':int}
    else:
        dtypes = {}
        dtypes[item_name] =  str
        for c in classes:
            dtypes[c] = int
    df_label = pd.read_csv(databasetxt,sep=",",dtype=dtypes)
    str_val = 'val'
    if database=='Wikidata_Paintings_miniset_verif':
        df_label = df_label[df_label['BadPhoto'] <= 0.0]
        str_val = 'validation'
    elif database=='Paintings':
        str_val = 'validation'
    elif database in ['VOC2007','watercolor','clipart','PeopleArt','CASPApaintings','comic','RASTA']:
        str_val = 'val'
        df_label[classes] = df_label[classes].apply(lambda x:(x + 1.0)/2.0)
        # To cast from -1 1 to 0.0 1.0
    
    return(item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC)