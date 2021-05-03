#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:27:29 2019

Lecture des fichiers de sorties de OIV5

@author: gonthier
"""

import glob 
import pandas as pd

from IMDB import get_database

default_path_imdb = '/media/gonthier/HDD2/data/'
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

path_to_metrics = '/media/gonthier/HDD2/data/OIV5/output/metrics/'

list_files = glob.glob(path_to_metrics+'*')

classes_descrip = pd.read_csv('/media/gonthier/HDD2/data/OIV5/challenge-2019-classes-description-500.csv',header=None)
classes_descrip = classes_descrip.append(pd.DataFrame([['item','mAP@0.5IOU']]))
classes_descrip.columns = ['ID','class']

list_lines = {}

topline = '\\begin{longtable}{c|c|c|c|'
list_line_name = ['Classe'] + classes + ['mAP@0.5IOU']
for l in ['Classe','mAP@0.5IOU']:
    list_lines[l] = [l]  
for l in classes:
    list_lines[l] = []    
    
for c in classes:
    cstr = classes_descrip[classes_descrip['ID']==c]['class'].values[0]
    cstr = cstr.replace('&','')
    list_lines[c] += cstr
    
list_lines['Classe'] +=  '& 3k & 33k & Test'

for database in ['OIV5_small_3135','OIV5_small_30001']:
    if 'OIV5_small_3135' ==database :
        num = 3135
    elif 'OIV5_small_30001' ==database:
        num = 30001
    dtypes = {}
    dtypes['item'] =  str
    for c in classes:
        dtypes[c] = int
    path_data_csvfile = '/media/gonthier/HDD2/data/OIV5/'
    databasetxt = path_data_csvfile + database + '.csv'
    df_label = pd.DataFrame(pd.read_csv(databasetxt,sep=",",dtype=dtypes))
    df_train = df_label[df_label['set']=='train']
    df_val = df_label[df_label['set']=='test']
    df_train.drop('set',axis=1,inplace=True)
    df_train.drop('item',axis=1,inplace=True)
    df_val.drop('set',axis=1,inplace=True)
    df_val.drop('item',axis=1,inplace=True)
    
    df = df_train.sum()
    dfval = df_val.sum()
    df['item'] = num
    for c in classes:
        list_lines[c] += '& ' +str(df[c])
    list_lines['mAP@0.5IOU'] += '& ' +str(df['item'])
    
for c in classes:
    list_lines[c] += '& ' +str(dfval[c])
list_lines['mAP@0.5IOU'] += '& ' +str(len(df_val))

for file in list_files:
    topline += 'c'
    if 'WRC' in file:
        with_scores = True
        case_s = ' S' 
    else:
        with_scores = False
        case_s = ''
    if 'OIV5_small_3135' in file:
        database ='OIV5_small_3135'
        short_name = '3k'
        num = 3135
    elif 'OIV5_small_30001' in file:
        database = 'OIV5_small_30001'
        short_name = '30k'
        num = 30001
    if 'MaxOfMax' in file:
        case_m = ' MoM'
    else:
        case_m = ''
    item_name,path_to_img,default_path_imdb,classes,ext,num_classes,str_val,df_label,path_data,Not_on_NicolasPC =\
        get_database(database)
    APs = pd.read_csv(file,sep=',',header=None,dtype={0:str,1:float})
    name_columns_AP = 'AP ' + short_name + case_s + case_m
    list_lines['Classe'] += ' & ' + short_name + case_s + case_m
    APs.columns = ['ID',name_columns_AP]
    APs[name_columns_AP] = APs[name_columns_AP].apply(lambda x: x*100)
    APs['ID'] = APs['ID'].apply(lambda x: x.replace('OpenImagesDetectionChallenge_Precision/',''))
    APs['ID'] = APs['ID'].apply(lambda x: x.replace("OpenImagesDetectionChallenge_PerformanceByCategory/AP@0.5IOU/b",''))
    APs['ID'] = APs['ID'].apply(lambda x: x.replace("'",''))
    for row in APs.iterrows():
        ap_c = row[1][1]
        c = row[1][0]
        if float(ap_c) > 5.0:
            list_lines[c] += "& \\textbf{{ {0:.1f} }}".format(ap_c)
        else:
            list_lines[c] += "& {0:.1f}".format(ap_c)

topline += '}'   

preambul = r'\documentclass[12pt]{report} \usepackage[english,french]{babel} \usepackage[top=2cm,bottom=1cm,left=1.0cm,right=1.0cm,marginparwidth=1.0cm]{geometry} \usepackage{longtable} \begin{document}'
title = r'\title{OIV5 detection score} \author{Nicolas Gonthier \\   LTCI, Télécom ParisTech, Université Paris Saclay,\\} \date{\today} \maketitle'

print(preambul)
print(title)
print(topline)
for l in list_line_name:
    list_lines[l] += ' \\\\'
    if l=='Classe':
        list_lines['Classe'] += ' \hline'
    print(''.join(list_lines[l]))
endline = '\end{longtable} \end{document}'
print(endline)


    

    
