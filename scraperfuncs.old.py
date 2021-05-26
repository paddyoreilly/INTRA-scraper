# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:36:15 2021

@author: paddy
"""

import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import dates
from urllib.request import urlopen, HTTPError
import gc
import tqdm

def scraper_day(y,m,d):
    """
    

    Parameters
    ----------
    y : int
        Year.
    m : int
        Month.
    d : int
        Day.

    Returns
    -------
    dayinfo : list
        List containing RSP info for the day from solarmonitor.org.

    """
            
    page = datetime.strftime(datetime(year=int(y),month=int(m),day=int(d)),
                             'https://solarmonitor.org/data/%Y/%m/%d/meta/noaa_events_raw_%Y%m%d.txt')
    problem = False
    dayinfo=[]
    
    try:
        urlopen(page)
    except HTTPError:
        #print('No data for: '+str(d)+'/'+str(m)+'/'+str(y))
        problem = True
    
    if problem == False:
        textfile = urlopen(page).read().decode('utf-8')
        events = textfile.splitlines()[13:]
        while '' in events:
            events.remove('')
            
        if len(events) >= 2:
            for i in events:
                if i[43:46] == 'RSP':
    
                    try: [datetime.strptime(str(i[11:15])+str(y)+str(m)+str(d), '%H%M%Y%m%d'),
                          datetime.strptime(str(i[28:32])+str(y)+str(m)+str(d), '%H%M%Y%m%d')]
                    except ValueError:
                        print('The following needs correcting:\n'+i)
                        a = input(str(i[11:15])+' start time: ')
                        b = input(str(i[28:32])+' end time: ')
                    else:
                        a, b = str(i[11:15]), str(i[28:32])
                        
                    dayinfo.append([[datetime.strptime(a+str(y)+str(m)+str(d), '%H%M%Y%m%d'),
                                               datetime.strptime(b+str(y)+str(m)+str(d), '%H%M%Y%m%d')],
                                              i[58:73]])
    
    return dayinfo

#%% Returns info for each day since 2017

def infogetter():
    day = datetime(year=2017, month=1, day=1)
    
    info={}
    
    while day < datetime.now():
        y, m, d = datetime.strftime(day, '%Y'), datetime.strftime(day, '%m'), datetime.strftime(day, '%d')
        
        temp = scraper_day(y,m,d)
        
        if len(temp) > 0: 
            info[day] = temp
            
        day=day+timedelta(days=1)
        
    return info
    
#%% Splits the RSP info into different burst types

def rspsplit(info):
    
    split = {'II':[],'III':[],'IV':[],'V':[],'VI':[],'VII':[],'CTM':[]}
    
    for i in info:
        for s in split:
            if s+'/' == i[1][:len(s)+1]:
                split[s] = split[s] + [i]
                break
            
    return split

#%% Splits the XRA into different burst types

def xrasplit(info):
    
    split = {'B':[],'C':[],'X':[],'M':[]}
    
    for i in info:
        for s in split:
            if s == i[1][0]:
                split[s] = split[s] + [i]
                break
            
    return split
    
#%% Makes an empty list of days since 2017

def daylistmaker():
    days=[]
    day = datetime(year=2017, month=1, day=1)
    while day < datetime.now():
        days.append(day)
        day=day+timedelta(days=1)
        
    return days

#%% Combines the RSP info and XRA info

def combiner(RSPinfo_split,XRAinfo_split):
    infocombined={}
    pre = {'B':[],'C':[],'X':[],'M':[],'II':[],'III':[],'IV':[],'V':[],'VI':[],'VII':[],'CTM':[]}
    daylist=daylistmaker()
    for day in daylist:
        if day in RSPinfo_split:
            infocombined[day] = pre.copy()
            infocombined[day].update(RSPinfo_split[day])
        if day in XRAinfo_split:
            if day in RSPinfo_split:
                infocombined[day].update(dict(**RSPinfo_split[day],**XRAinfo_split[day]))
            else:
                infocombined[day] = pre.copy()
                infocombined[day].update(XRAinfo_split[day])

#%% Renamer and deleter if the info dictionaries need editing

def renamer(dictionary,before,after):
    for day in dictionary:
        try: dictionary[day][before]
        except KeyError:
            continue
        dictionary[day][after] = dictionary[day][before]
        del dictionary[day][before]

def deleter(dictionary,item):
    for day in dictionary:
        if item in dictionary[day]:
            del dictionary[day][item]
            
#%% Plots the Spectrogram alongside the SWPC info

def burstplot(bstfile, graphinfo, saveloc):
    """
    Plots the spectrum along side a catalog of the radio bursts

    Parameters
    ----------
    bstfile : .dat file
        The I-LOFAR bst .dat file.
    graphinfo : dict
        Dictionary in the structure; day: burst type: [start time, end time].

    Returns
    -------
    Plot of the spectrum along side a catalog of the radio bursts

    """
    def start(bstfile, subbands=np.arange(488)):
        def sb_to_freq(sb=np.arange(488)):
            def sb_to_freq_math(x): return ((n-1)+(x/512))*(clock/2)
            clock = 200 #MHz
            sb_3 = np.arange(54,454,2)
            sb_5 = np.arange(54,454,2)
            sb_7 = np.arange(54,290,2)
            n = 1
            freq_3 = sb_to_freq_math(sb_3)
            n = 2
            freq_5 = sb_to_freq_math(sb_5)
            n = 3
            freq_7 = sb_to_freq_math(sb_7)
            freq = np.concatenate((freq_3,freq_5,freq_7),axis=0)
            freq = freq[sb[0]:sb[-1]+1]
            return freq   
    
        data = np.fromfile(bstfile)
        
        file_size = os.path.getsize(bstfile) 
        bit_mode = round(file_size/data.shape[0])
        t_len = (data.shape[0]/len(subbands))
        data = data.reshape(-1,len(subbands))
        
        global obs_start
        obs_start = bstfile[len(bstfile)-27:len(bstfile)-12]
        
        obs_start = datetime.strptime(obs_start,"%Y%m%d_%H%M%S")
        global date_format
        date_format = dates.DateFormatter("%H:%M")
        obs_len = timedelta(seconds = t_len)
        obs_end = obs_start + obs_len
        t_lims = [obs_start, obs_end]
        t_lims = dates.date2num(t_lims)
        
        #time array
        
        global t_arr 
        t_arr = np.arange(0,data.shape[0])
        t_arr = t_arr*timedelta(seconds=1)
        t_arr = obs_start+t_arr
        t_arr = dates.date2num(t_arr)
        
        global data_F 
        data_F = data/np.mean(data[:100],axis=0)
        sbs = subbands
        global freqs
        freqs = sb_to_freq(sbs)

    def pcolormeshplot(data, y, x, vdata=None, tfs=True, colorbar=False,cmap=cmocean.cm.solar,alpha=None,figsize=None, flip=False):
    
        if vdata is None:
            vdata = data
        g1 = 200
        g2 = 400
            
        if figsize is None:
            pass
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=400)
        
        plt.pcolormesh(x,y[:g1],data.T[:g1], shading='auto',
                       vmin=np.percentile(vdata,2), vmax=np.percentile(vdata,98),cmap=cmap,alpha=alpha);
        if tfs == True:
            plt.pcolormesh(x,y[g1:g2],data.T[g1:g2], shading='auto',
                           vmin=np.percentile(vdata,2), vmax=np.percentile(vdata,98),cmap=cmap,alpha=alpha);
            plt.pcolormesh(x,y[g2:],data.T[g2:], shading='auto',
                           vmin=np.percentile(vdata,2), vmax=np.percentile(vdata,98),cmap=cmap,alpha=alpha);
        
        if flip == True:
            plt.gca().invert_yaxis()
            
        plt.gca().xaxis_date()
        plt.gca().xaxis.set_major_formatter(date_format)
        
        plt.xlabel("Time")
        plt.ylabel("Frequency (MHz)")
        
        if colorbar == True:
            plt.colorbar()

    start(bstfile)
    bstfile = bstfile[-27:]
    day = datetime(year =  int(datetime.strftime(obs_start, '%Y')),
                   month = int(datetime.strftime(obs_start, '%m')),
                   day  =  int(datetime.strftime(obs_start, '%d')))
    coltyp = {'Type B':'black','Type C':'black','Type X':'black','Type M':'black',
              'Type II':'yellow','Type III':'blue','Type IV':'red','Type V':'brown',
              'Type VI':'green','Type VII':'purple','Type CTM':'pink'}
    typorder = ['Type B', 'Type C',  'Type X', 'Type M',
                'Type II','Type III','Type IV','Type V','Type VI','Type VII','Type CTM']
    
    fig, ax = plt.subplots(figsize=(10,15), dpi=200)
    
    plt.subplot2grid((5,1), (0,0))
    
    try: graphinfo[day]
    except KeyError:
        for typ in typorder:
            plt.barh(typ,[0])

    else:
        for typ in typorder:
            try: graphinfo[day][typ]
            except KeyError:
                plt.barh(typ,[0])
            else:
                if len(graphinfo[day][typ])==0:
                    plt.barh(typ,[0])
                    
                else:
                    widths=[]
                    bots=[]
                    for burst in graphinfo[day][typ]:
                        if burst[0][1]-burst[0][0] < timedelta(minutes=2):
                            widths.append(timedelta(minutes=2))
                        else:
                            widths.append(burst[0][1]-burst[0][0])
                        bots.append(burst[0][0])
                            
                    plt.barh(typ, widths, left=bots, color=coltyp[typ])
                
    plt.ylabel('X-RAY           RADIO   ')
    plt.gca().xaxis.set_visible(False)
    plt.xlim(t_arr[0], t_arr[-1])
    plt.title('SWPC Classifications')
    
    
    plt.subplot2grid((5,1), (1,0), rowspan=4)
    
    pcolormeshplot(data_F,freqs,t_arr, flip=True)
    plt.title('I-LOFAR Spectrogram')
    plt.xlim(t_arr[0], t_arr[-1])
    
    #plt.tight_layout()
    plt.suptitle(bstfile)
    plt.savefig(saveloc+bstfile[:-4]+'.png')
    plt.close()
    
#%% Plots data from an input folder into an output folder

def folderplotter(inputfolder, outputfolder, plotinfo):
    for filename in os.listdir(inputfolder):
        burstplot(inputfolder+'/'+filename, plotinfo ,outputfolder+'/')

#%% Will plot a from a list of given I-LOFAR urls. Also now outputs the SWPC textfile next to the plot output

import urllib3
import gc
import tqdm

def downloadplotter(urls,rsps_split):
    structure = 'D:/Users/paddy/Desktop/plotstructure/'
    tempdir = structure+'temp/'
    connection_pool = urllib3.PoolManager()
    
    if os.path.exists(tempdir):
        pass
    else:
        os.makedirs(tempdir)
        
    for url in tqdm.tqdm(urls):
        
        saveloc = structure + url[22:-27]
        if os.path.exists(saveloc + url[-27:-4] + '.png'):
            continue
        
        r = connection_pool.request('GET', url, preload_content=False)
        if int(r.headers['Content-Length'])%(8*488) != 0:
            print('\nValue Error')
            continue
        
        
        if os.path.exists(saveloc):
            pass
        else:
            os.makedirs(saveloc)
        bstloc = tempdir + url[-27:]
        
        with open(bstloc,'wb') as out:
            out.write(r.data)
        r.release_conn()
        
        burstplot(bstloc, rsps_split, saveloc)
        plt.close()        
        
        out.close()
        os.remove(bstloc)
        
        y, m, d = url[-27:-23], url[-23:-21], url[-21:-19]
        try: connection_pool.request('GET','https://solarmonitor.org/data/'+y+'/'+m+'/'+d+'/meta/noaa_events_raw_'+y+m+d+'.txt')
        except urllib3.HTTPError:
            txtcontent = 'Error: https://solarmonitor.org/data/'+y+'/'+m+'/'+d+'/meta/noaa_events_raw_'+y+m+d+'.txt'
        else:
            txtpage = connection_pool.request('GET','https://solarmonitor.org/data/'+y+'/'+m+'/'+d+'/meta/noaa_events_raw_'+y+m+d+'.txt')
            txtcontent = txtpage.data.decode('utf-8')
            
        
        with open(saveloc+y+m+d+'SWPC.txt','w+') as txtfile:
            txtfile.write(txtcontent)
        txtpage.release_conn()
            
        del txtcontent
                
        del saveloc ; del url ; del bstloc
        gc.collect()

#%% Below is ran to get an 'infocombined' dictionary compatable with downloadplotter() function

info = infogetter()

RSPinfo={}
XRAinfo={}

for day in info:
    if 'RSP' in info[day]:
        RSPinfo[day] = info[day]['RSP']
    if 'XRA' in info[day]:
        XRAinfo[day] = info[day]['XRA']

RSPinfo_split={}
for day in RSPinfo:
    RSPinfo_split[day] = rspsplit(RSPinfo[day])
    
XRAinfo_split={}
for day in XRAinfo:
    XRAinfo_split[day] = xrasplit(XRAinfo[day])
    
infocombined = combiner(RSPinfo_split, XRAinfo_split)

renamelist = { 'Type B'   : 'B'  ,
               'Type C'   : 'C'  ,
               'Type X'   : 'X'  ,
               'Type M'   : 'M'  ,
               'Type II'  : 'II' ,
               'Type III' : 'III',
               'Type IV'  : 'IV' ,
               'Type V'   : 'V'  ,
               'Type VI'  : 'VI' ,
               'Type VII' : 'VII',
               'Type CTM' : 'CTM'  }

for after in renamelist:
    renamer(infocombined,renamelist[after],after)
