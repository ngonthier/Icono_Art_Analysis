# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:04:51 2019

The goal of this script is to output Latex Results pdf 

@author: gonthier
"""

import os,glob
import subprocess
from subprocess import TimeoutExpired
import pickle
from MILbenchmark.utils import getClassesNames
import pathlib

def ProducePDF():

    list_datasets = ['Birds','SIVAL','Newsgroups']
    
    script_dir = os.path.dirname(__file__)
    path_file_results = os.path.join(script_dir,'MILbenchmark','Results','*.pkl')
    nameTex = 'Results.tex'
    path_file_tex = os.path.join(script_dir,'MILbenchmark','Results','pdf')
    pathlib.Path(path_file_tex).mkdir(parents=True, exist_ok=True)
    path_file_tex = os.path.join(path_file_tex,nameTex)
    dictOfSels = {}
    for dataset in list_datasets:
        dictOfSels[dataset] = {}
    for fname in glob.glob(path_file_results):
        head,tail = os.path.split(fname)
        tail_splitted = tail.split('.')[0]
        try:
            results = pickle.load(open(fname,'br'))
            for dataset in list_datasets:
                if dataset in tail_splitted:
                    print(tail_splitted)
                    dictOfSels[dataset][tail_splitted] = results
        except FileNotFoundError:
            print('Error with the file : ',fname)
        
            
            
    header = r'''\documentclass{article}
    \usepackage{graphicx}
    
    \begin{document}
    
    \begin{center}
    '''
    footer = r'''\end{center}
    \end{document}
    '''
    
    main = ''
    
    listOFmetrics = ['UAR','F1','AUC']
    listOFmetrics_index = [1,0,2]
    
    for dataset in sorted(dictOfSels.keys()):
        list_classes = getClassesNames(dataset)
        methods = dictOfSels[dataset].keys()
        print(methods)
        methods0 = list(methods)[0]
        print(methods0)
        for metric,indexmetric in zip(listOFmetrics,listOFmetrics_index):
            main += '\\begin{table}[h!] \n'
            captionstr = '\\caption{\\label{tab'+dataset+metric+'} '+ dataset+' ' + metric+' (\%)} \n'
            main += captionstr
            main +='\\resizebox{\columnwidth}{!}{ \n'
            main += '\\begin{tabular}{|c'
            for _ in methods:
                main += '|c'
            main +='|} \\hline  \n '
            lines_of_tab = []
            line0 = 'Method :'
            for classe in list_classes:
                classeET = classe.replace('_','\\_') 
                lines_of_tab +=[classeET]
            for method in sorted(methods):
                dataset_Underscore = '_' + dataset
                method_modified  = ' & ' + method.replace(dataset_Underscore,'')
                method_modified = method_modified.replace('_','\\_')
    #            method_modified  = method_modified.replace(dataset_Underscore,'')
                line0 += method_modified
                for i,classe in enumerate(list_classes):
                    try:
                        perf,_ = dictOfSels[dataset][method][classe]
                        print(perf[0])
                        print(perf[1])
                        mPerf = perf[0]
                        stdPerf = perf[1]
                        metricvalue = 100*mPerf[indexmetric]
                        metricvaluestd = 100*stdPerf[indexmetric]
                        str_print = "& {0:.2f} ({1:.2f}) ".format(metricvalue,metricvaluestd)
                    except KeyError:
                        str_print = " &"
                    lines_of_tab[i] += str_print
            line0 += ' \\\ \\hline \n'
            main += line0
            for i,classe in enumerate(list_classes):
                str_print = " \\\ \n"
                lines_of_tab[i] += str_print
                main +=  lines_of_tab[i]
            main = main + '\\hline  \n \\end{tabular} \n  } \\end{table} '
    
    print('End with the creation of the tabulars')
    content = header + main + footer
    
    with open(path_file_tex,'w') as f:
        f.write(content)
    
    
#    print('Start the pdf compiling')
#    commandLine = subprocess.Popen(['pdflatex', 'yields.tex'])
#    try:
#        outs, errs = commandLine.communicate(timeout=1)
#        os.unlink('yields.tex')
#        os.unlink('yields.log')
#        os.unlink('yields.aux')
#    except TimeoutExpired:
#        commandLine.kill()
#        print('Kill the process due to timeout')
#    print('Finish the pdf compiling')
    


if __name__ == '__main__':
    ProducePDF()