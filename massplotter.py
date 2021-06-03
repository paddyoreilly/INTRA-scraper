# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:25:01 2021

@author: paddy
"""

import os


import numpy as np
import matplotlib.pyplot as plt
import urllib3
from urllib.request import HTTPError

from datetime import datetime, timedelta
from astropy.time import Time
from matplotlib import dates, cm
from matplotlib.ticker import MultipleLocator
import json
from astropy.io import fits

import gc
from tqdm import tqdm
from time import sleep
from bst_newplotter import plot
from spyder_kernels.utils import iofuncs

def massplotter():
    
    urls = iofuncs.load_dictionary('C:/Users/paddy/OneDrive/DCU/ilofar/newwork/scrape/urlforplot.spydata')[0]['urls']
    
    def removeandwait(url):
        urls.remove(url)
        sleep(5)
        
    """
    
    Parameters
    ----------
    urls : List
    List of I-LOFAR '.bst' urls that will be plotted

    Returns
    -------
    '.png' plots alongs with '.txt' text files in the same file format as the urls.
    
    """
    
    structure = 'D:/Users/paddy/Desktop/plotstructure/'
    tempdir = structure+'temp/'    

    
    
    if os.path.exists(tempdir):
        pass
    else:
        os.makedirs(tempdir)
    outof=str(len(urls))
    prog=0
    for url in urls:
        
        prog=prog+1
        print('\n================================================\n' + str(prog)+'/'+outof+'\n')
        
        saveloc = structure + url[22:-27]
        if os.path.exists(saveloc + url[-27:-4] + '.png'):
            print('\nPlot already exists\n' + saveloc + url[-27:-4] + '.png')
            continue
          
        bstloc = tempdir + url[-27:]
        
        print('Connecting...')
        connection_pool = urllib3.PoolManager()
        r = connection_pool.request('GET', url)
        print('Connection Got\n')
        
        filesize = r.headers['Content-Length']
        
        if filesize == '284':
            print('\nError getting file\nTrying again\n')
            tries=0
            while filesize == '284':
                print('Waiting 10 seconds\n')
                sleep(10)
                r = connection_pool.request('GET', url)
                filesize = r.headers['Content-Length']
                tries = tries+1
                if tries > 3:
                    print('\nToo many tries\nMoving on\n')
                    filesize = '284'
                else:
                    print('No luck\n')
                
        if int(filesize)%(8*488) != 0:
            print('\nFile Size Error: ' + str(filesize) + ' bytes\n' + url)
            sleep(5)
            continue
        
        with open(bstloc,'wb') as out:
            out.write(r.data)
            
        r.release_conn()
        
        if os.path.exists(saveloc) is False:
            os.makedirs(saveloc)
        
        print('\nBegin Plot:')
        plot(bstloc, saveloc)
        print('\nPlotted\n' + saveloc + url[-27:-4] + '.png')
        sleep(2)
        
        os.remove(bstloc)
    
if __name__ == '__main__':
    massplotter()
