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
import numpy as np
from scipy.stats import ttest_ind_from_stats


def ProduceTexTabular():
    
    list_datasets = ['Birds','SIVAL','Newsgroups']
    script_dir = os.path.dirname(__file__)
    path_file_results = os.path.join(script_dir,'MILbenchmark','Results','*.pkl')
    
    dictOfSels = get_MIL_benchmarck_results(list_datasets)

    listOFmetrics = ['UAR','F1','AUC']
    listOFmetrics_index = [1,0,2]

    for dataset in sorted(dictOfSels.keys()):
        if dataset in ['Newsgroups', 'SIVAL']:
            list_classes = sorted(getClassesNames(dataset))
        else:
            list_classes = getClassesNames(dataset)
        methods = dictOfSels[dataset].keys()
        
        dict_list_metric_value = {}
        dict_list_std_metric_value = {}
        
        # One table per metric 
        for metric,indexmetric in zip(listOFmetrics,listOFmetrics_index):
            
            for i,classe in enumerate(list_classes):
                # First determine the best of the line for a given class
                list_metric_value = []
                list_std_metric_value = []
                for method in sorted(methods):
                    try:
                        perf,_ = dictOfSels[dataset][method][classe]
                        mPerf = perf[0]
                        stdPerf = perf[1]
                        metricvalue = 100*mPerf[indexmetric]
                        metricvaluestd = 100*stdPerf[indexmetric]
                        list_metric_value += [metricvalue]
                        list_std_metric_value += [metricvaluestd]
                    except KeyError:
                       list_metric_value += [-1.]
                       list_std_metric_value += [0.]
            dict_list_metric_value[metric] = list_metric_value
            dict_list_std_metric_value[metric] = list_std_metric_value
                
        
        
#        print(methods)
        main =''
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
                
                
            list_wins = [0]*len(methods)
            i_method = 0
            # This loop fct will prepare the line0 (name of the methods)
            # But also the line of the performance for each class
            for method in sorted(methods):
                dataset_Underscore = '_' + dataset
                method_modified  = ' & ' + method.replace(dataset_Underscore,'')
                method_modified = method_modified.replace('_','\\_')
    #            method_modified  = method_modified.replace(dataset_Underscore,'')
                line0 += method_modified
                print(sorted(list_classes))
                for i,classe in enumerate(list_classes):
                    
                    list_metric_value = dict_list_metric_value[metric]
                    list_metric_value_i = list_metric_value[i]
                    list_std_metric_value = dict_list_std_metric_value[metric]
                    list_std_metric_value_i = list_std_metric_value[i]
                    max_on_class_i = np.max(list_metric_value_i)
                    print(np.amax(list_metric_value_i))
                    print(list_std_metric_value_i)
                    std_max_on_class_i = list_std_metric_value_i[np.amax(list_metric_value_i)]
                    print(i,classe,max_on_class_i,std_max_on_class_i)
                    
                    try:
                        perf,_ = dictOfSels[dataset][method][classe]
#                        print(perf[0])
#                        print(perf[1])
                        mPerf = perf[0]
                        stdPerf = perf[1]
                        metricvalue = 100*mPerf[indexmetric]
                        metricvaluestd = 100*stdPerf[indexmetric]
                        s,pv = ttest_ind_from_stats(mPerf,stdPerf, 10, max_on_class_i,
                                                    std_max_on_class_i, 10,
                                                    equal_var=False)
                        if pv >= 0.5:
                            str_print = "& \textbf{ {0:.2f} } ({1:.2f}) ".format(metricvalue,metricvaluestd)
                            list_wins[i_method] += 1
                        else:                        
                            str_print = "& {0:.2f} ({1:.2f}) ".format(metricvalue,metricvaluestd)
                    except KeyError:
                        str_print = " &"
                    lines_of_tab[i] += str_print
                i_method += 1
            line0 += ' \\\ \\hline \n'
            main += line0
            
            # Line with the winning per methods
            line1 = ' Wins '+ metric+ ' & ' 
            for wins in list_wins:
                line1 += str(wins) + ' & '
            line1 += ' \\\ \\hline \n'
            main += line1
            
            for i,classe in enumerate(list_classes):
                str_print = " \\\ \n"
                lines_of_tab[i] += str_print
                main +=  lines_of_tab[i]
            main = main + '\\hline  \n \\end{tabular} \n  } \\end{table} '
    
    print('End with the creation of the tabulars')
    print(main)


def get_MIL_benchmarck_results(list_datasets):

    script_dir = os.path.dirname(__file__)
    path_file_results = os.path.join(script_dir,'MILbenchmark','Results','*.pkl')
#    nameTex = 'ResultsMI.tex'
#    path_file_tex = os.path.join(script_dir,'MILbenchmark','Results','pdf')
#    pathlib.Path(path_file_tex).mkdir(parents=True, exist_ok=True)
#    path_file_tex = os.path.join(path_file_tex,nameTex)
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
            
    return(dictOfSels)
    
def ProducePDF():
    

    list_datasets = ['Birds','SIVAL','Newsgroups']
    dictOfSels = get_MIL_benchmarck_results(list_datasets)
    
        
            
            
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
    
    print(sorted(dictOfSels.keys()))
    
    for dataset in sorted(dictOfSels.keys()):
        if dataset in ['Newsgroups', 'SIVAL']:
            list_classes = sorted(getClassesNames(dataset))
        else:
            list_classes = getClassesNames(dataset)
        methods = dictOfSels[dataset].keys()
#        print(methods)
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
                print(sorted(list_classes))
                for i,classe in enumerate(list_classes):
                    try:
                        perf,_ = dictOfSels[dataset][method][classe]
#                        print(perf[0])
#                        print(perf[1])
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
    #ProducePDF()
    ProduceTexTabular()