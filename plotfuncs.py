# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:22 2021

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
import tqdm

     


class plot:
    
    def gridmaker(obs_start, t_arr):
        
        bargrid=[]
        
        temptime = datetime(year  = int(datetime.strftime(obs_start, '%Y')),
                            month = int(datetime.strftime(obs_start, '%m')),
                            day   = int(datetime.strftime(obs_start, '%d')),
                            hour  = int(datetime.strftime(obs_start, '%H')))
        while temptime < obs_start + timedelta(seconds=len(t_arr)):
            bargrid.append(temptime)
            temptime=temptime+timedelta(hours=1)
            
        for hour in bargrid:
            plt.axvline(hour, c='black', alpha=0.1)
        del temptime
    
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
        
        obs_start = bstfile[len(bstfile)-27:len(bstfile)-12]
        
        obs_start = datetime.strptime(obs_start,"%Y%m%d_%H%M%S")

        date_format = dates.DateFormatter("%H:%M")
        obs_len = timedelta(seconds = t_len)
        obs_end = obs_start + obs_len
        t_lims = [obs_start, obs_end]
        t_lims = dates.date2num(t_lims)
        
        
        t_arr = np.arange(0,data.shape[0])
        t_arr = t_arr*timedelta(seconds=1)
        t_arr = obs_start+t_arr
        t_arr = dates.date2num(t_arr)

        data_F = data/np.mean(data[:100],axis=0)
        sbs = subbands

        freqs = sb_to_freq(sbs)
        
        return obs_start, date_format, t_arr, data_F, freqs
    
    def spectrogram(data, y, x, date_format, bstfile, vdata=None, tfs=True, colorbar=False, cmap=cm.Spectral_r, alpha=None, figsize=None, flip=True):
        
        if vdata is None:
            vdata = data
        g1 = 200
        g2 = 400
            
        if figsize is None:
            pass
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=400)
        
        v1 , v2 = 15 , 98
        
        plt.pcolormesh(x,y[:g1],data.T[:g1], shading='auto',
                       vmin=np.percentile(vdata,v1), vmax=np.percentile(vdata,v2),cmap=cmap,alpha=alpha);
        if tfs == True:
            plt.pcolormesh(x,y[g1:g2],data.T[g1:g2], shading='auto',
                           vmin=np.percentile(vdata,v1), vmax=np.percentile(vdata,v2),cmap=cmap,alpha=alpha);
            plt.pcolormesh(x,y[g2:],data.T[g2:], shading='auto',
                           vmin=np.percentile(vdata,v1), vmax=np.percentile(vdata,v2),cmap=cmap,alpha=alpha);
        
        if flip == True:
            plt.gca().invert_yaxis()
            
        plt.gca().xaxis_date()
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.gca().yaxis.set_major_locator(MultipleLocator(20))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(10))
        
        plt.xlabel("UTC "+datetime.strftime(dates.num2date(x[0]), '(%d-%b-%Y)'))
        plt.ylabel("Frequency (MHz)")
        
        if colorbar == True:
            plt.colorbar()
            
        plt.title('I-LOFAR Spectrogram: '+bstfile)
        plt.xlim(x[0], x[-1])

    def classes(page, t_arr):
        
        def infofrompage(page):
            def rspsplit(info):
                split = {'II':[],'III':[],'IV':[],'V':[],'VI':[],'VII':[],'CTM':[]}
                for i in info:
                    for s in split:
                        if s+'/' == i[1][:len(s)+1]:
                            split[s] = split[s] + [i]
                            break
                return split
            def xrasplit(info):
                split = {'C':[],'X':[],'M':[]}
                for i in info:
                    for s in split:
                        if s == i[1][0]:
                            split[s] = split[s] + [i]
                            break
                return split
            def renamer(dictionary,before,after):
                try: dictionary[before]
                except KeyError:
                    print('error: '+dictionary+', '+before+', '+after)
                else:
                    dictionary[after] = dictionary[before]
                    del dictionary[before]
        
        
            y, m, d = int(page[-12:-8]), int(page[-8:-6]), int(page[-6:-4])
        
        
            dayinfo=[]
        
            text = urllib3.PoolManager().request('GET',page).data.decode()
            events = text.splitlines()[13:]
            while '' in events:
                events.remove('')
                
            if len(events) >= 2:
                for i in events:
                    if i[43:46] == 'RSP' or i[43:46] == 'XRA':
        
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
            
            rsp =     rspsplit(dayinfo)
            dayinfo = xrasplit(dayinfo)
            
            dayinfo.update(rsp)
            
            renamelist = {'C Class'   : 'C'  ,
                          'X Class'   : 'X'  ,
                          'M Class'   : 'M'  ,
                          'Type II'  : 'II' ,
                          'Type III' : 'III',
                          'Type IV'  : 'IV' ,
                          'Type V'   : 'V'  ,
                          'Type VI'  : 'VI' ,
                          'Type VII' : 'VII',
                          'Type CTM' : 'CTM'}
            
            for after in renamelist:
                renamer(dayinfo,renamelist[after],after)
            
            return dayinfo
        
        info = infofrompage(page)
        typorder = ['C Class',  'M Class', 'X Class',
                    'Type II','Type III','Type IV','Type V','Type VI','Type VII','Type CTM']
        
        for typ in typorder:
            widths=[]
            lefts=[]
            
            if len(info[typ]) == 0:
                plt.barh(typ,[0])
                
            else:                
                for burst in info[typ]:
                    
                    if burst[0][1]-burst[0][0] < timedelta(minutes=2):
                        widths.append(timedelta(minutes=2))
                        lefts.append(burst[0][0]+timedelta(minutes=1))
                    
                    else:
                        widths.append(burst[0][1]-burst[0][0])
                        lefts.append(burst[0][0])
                        
            plt.barh(typ, widths, left=lefts, color='black', alpha=0.6)
            
            
        plt.ylabel('X-RAY              RADIO        ')
        plt.gca().xaxis.set_visible(False)
        plt.xlim(t_arr[0], t_arr[-1])
        plt.title('NOAA/SWPC Event Classifications')
        plt.gca().yaxis.set_label_position("right")
        
        for i in range(int(len(typorder)/2)):
            plt.axhline(int(i*2),c='black',linewidth=17.5, alpha=0.05)
        
    def goes(tstart, tend):
      
        def smooth(x,window_len=11,window='hanning'):
            
            """smooth the data using a window with requested size.
        
            This method is based on the convolution of a scaled window with the signal.
            The signal is prepared by introducing reflected copies of the signal
            (with the window size) in both ends so that transient parts are minimized
            in the begining and end part of the output signal.
        
            input:
                x: the input signal
                window_len: the dimension of the smoothing window; should be an odd integer
                window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.
        
            output:
                the smoothed signal
        
            example:
        
            t=linspace(-2,2,0.1)
            x=sin(t)+randn(len(t))*0.1
            y=smooth(x)
        
            see also:
        
            numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
            scipy.signal.lfilter
        
            NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
            """
        
            if x.ndim != 1:
                raise ValueError; "smooth only accepts 1 dimension arrays."
        
            if x.size < window_len:
                raise ValueError; "Input vector needs to be bigger than window size."
        
        
            if window_len<3:
                return x
        
        
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError; "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        
        
            s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
            #print(len(s))
            if window == 'flat': #moving average
                w = np.ones(window_len,'d')
            else:
                w = eval('np.'+window+'(window_len)')
        
            y = np.convolve(w/w.sum(), s, mode='valid')
            return y

        def get_goes(day):
            
            daycheck = Time(day)
            if daycheck > datetime.now()-timedelta(days=6):
                
                return get_goesnew(url='https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json')
                
            else:
                return get_goesarchive(day)

        def get_goesnew(url='https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json'):
            ''' Read a single one of the standard GOES files (determined by the value of the
                url string.)
                
                The routine reads the json file pointed to by the url, and replaces any zeros with nan.
                
                Returns arrays of the GOES low energy (1-8 A) flux, the GOES high-energy (0.5-4 A) flux,
                and a Time object that is the array of UT times.
            '''
            f = urllib3.PoolManager().request('GET',url)
            txt = f.data
            goes = json.loads(txt)
            goeshi = []
            goeslo = []
            goestime = []
            for i in goes:
                if i['energy'] == '0.05-0.4nm':
                    goeshi.append(i['flux'])
                    goestime.append(i['time_tag'])
                else:
                    goeslo.append(i['flux'])
            goeslo = np.array(goeslo)
            goeslo[np.where(goeslo == 0.0)] = np.nan
            goeshi = np.array(goeshi)
            goeshi[np.where(goeshi == 0.0)] = np.nan
            if len(goestime) == 0:
                return [], [], []
            return goeslo, goeshi, Time(goestime), url
        
        def get_goesarchive(day):
    
            goesurl = datetime.strftime(day,'https://umbra.nascom.nasa.gov/goes/fits/%Y/go15%Y%m%d.fits')
            
            sats = ['15','13','14','16']
            for sat in sats:
                goesurl.replace(goesurl[45:47], sat)
                try: fits.open(goesurl)
                except HTTPError:
                    if sat == sats[-1]:
                        return [], [], [], 'URL Error'
                else: break
            
            goesdata = fits.open(goesurl)
            
            goesdata = goesdata[2].data
            
            goeslo   = goesdata['Flux'][0,0:,0]
            goeshi   = goesdata['Flux'][0,0:,1]
            goestimenums = goesdata['Time'].T[0:,0]
            
            daystart = datetime.strptime(datetime.strftime(day, '%Y%m%d'), '%Y%m%d')
            
            goestime=[]
            for s in goestimenums:
                goestime.append(daystart + timedelta(seconds = s))
                
            return goeslo, goeshi, Time(goestime), goesurl
        
        
        classes = ['A','B','C','M','X']
        
        plt.gca().xaxis_date()
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
        
        tstart = dates.num2date(tstart)
        tend = dates.num2date(tend)
        
        if tend - tstart >= timedelta(hours=5):
            smth = True
        else: smth = False
        
        lo, hi, t, goesurl = get_goes(tstart)
        
        tstart = Time(tstart)
        tend = Time(tend)
        
        if len(t) > 0:
        
            if smth == True:
                hi = smooth(hi,20,'blackman')[10:-9]
                lo = smooth(lo,20,'blackman')[10:-9]
                
            plt.gca().plot_date(t.plot_date,lo,'-',label='  1-8 A')
            plt.gca().plot_date(t.plot_date,hi,'-',label='0.5-4 A')
            
                            
            plt.xticks(tstart.plot_date + np.arange(7)/24.)
            plt.gca().xaxis.set_visible(False)
            plt.gca().set_title('GOES SXR Plot')
            
        plt.gca().set_ylim(1e-9,1e-2)
        
        for i in range(5):
            plt.gca().text(1.01,1/7.*i + 1.4/7,classes[i],transform=plt.gca().transAxes)
        
        plt.gca().set_ylabel(r'SXR Flux [W m$\mathregular{^{-2}}$]',fontsize=14)
        plt.gca().set_xlim(tstart.plot_date,tend.plot_date)
        plt.gca().set_yscale('log')
        plt.gca().legend()
        plt.gca().grid(axis='y',alpha=0.2, linestyle='--')
        plt.text(0.005, 0.99,goesurl, ha='left', va='top', transform=plt.gca().transAxes, fontsize=8)

def burstplot(bstfile, saveloc):
    """
    Plots the spectrum along side a catalog of the radio bursts

    Parameters
    ----------
    bstfile : .dat file
        The I-LOFAR bst .dat file.

    Returns
    -------
    Plot of the spectrum along side a catalog of the radio bursts

    """
    
    
    obs_start, date_format, t_arr, data_F, freqs = plot.start(bstfile)
    bstfile = bstfile[-27:]
    
    y, m, d = datetime.strftime(obs_start, '%Y'), datetime.strftime(obs_start, '%m'), datetime.strftime(obs_start, '%d')
    
    obsday = datetime(year =  int(y),
                   month = int(m),
                   day  =  int(d))
    
    swpcpage = 'https://solarmonitor.org/data/'+y+'/'+m+'/'+d+'/meta/noaa_events_raw_'+y+m+d+'.txt'
    
    fig, ax = plt.subplots(figsize=(10,15), dpi=200)
    
    plt.subplot2grid((5,1), (0,0))
    plot.classes(swpcpage, t_arr)
    plot.gridmaker(obs_start, t_arr)
    
    
    plt.subplot2grid((5,1), (1,0))
    plot.goes(t_arr[0], t_arr[-1])
    plot.gridmaker(obs_start, t_arr)
    
    plt.subplot2grid((5,1), (2,0), rowspan=3)
    plot.spectrogram(data_F,freqs,t_arr, date_format, bstfile)
    plot.gridmaker(obs_start, t_arr)
    

    plt.suptitle('Created: '+datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M')+' ',x=1,y=0,ha='right',va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(saveloc+bstfile[:-4]+'.png')
    plt.close()
    


def massplotter(urls):
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
    connection_pool = urllib3.PoolManager()
     
        

    
    
    if os.path.exists(tempdir):
        pass
    else:
        os.makedirs(tempdir)
        
    for url in tqdm.tqdm(urls):
        
        saveloc = structure + url[22:-27]
        if os.path.exists(saveloc + url[-27:-4] + '.png'):
            urls.remove(url)
            continue
        
        r = connection_pool.request('GET', url)

        filesize = r.headers['Content-Length']
        if int(filesize)%(8*488) != 0:
            print('\nFile Size Error: ' + filesize + ' bytes\n' + url[-27:-4] + '.png')
            urls.remove(url)
            continue
        
        
        if os.path.exists(saveloc):
            pass
        else:
            os.makedirs(saveloc)
        bstloc = tempdir + url[-27:]
        
        with open(bstloc,'wb') as out:
            out.write(r.data)
        r.release_conn()
        
        burstplot(bstloc, saveloc)
        plt.close()        
        
        out.close()
        os.remove(bstloc)
        
        y, m, d = url[-27:-23], url[-23:-21], url[-21:-19]
    
        if os.path.exists(saveloc+y+m+d+'SWPC.txt'):
            pass
        else:
            try: connection_pool.request('GET','https://solarmonitor.org/data/'+y+'/'+m+'/'+d+'/meta/noaa_events_raw_'+y+m+d+'.txt')
            except urllib3.HTTPError:
                txtcontent = 'Error: https://solarmonitor.org/data/'+y+'/'+m+'/'+d+'/meta/noaa_events_raw_'+y+m+d+'.txt'
                print('Error: https://solarmonitor.org/data/'+y+'/'+m+'/'+d+'/meta/noaa_events_raw_'+y+m+d+'.txt')
            else:
                txtpage = connection_pool.request('GET','https://solarmonitor.org/data/'+y+'/'+m+'/'+d+'/meta/noaa_events_raw_'+y+m+d+'.txt')
                txtcontent = txtpage.data.decode('utf-8')
            
            with open(saveloc+y+m+d+'SWPC.txt','w+') as txtfile:
                txtfile.write(txtcontent)
                    
            txtpage.release_conn()
        
            del txtcontent
            urls.remove(url)
        
        del saveloc ; del url ; del bstloc
        gc.collect()
        
    os.remove(tempdir)
    
def folderplotter(inputfolder, outputfolder):
    for filename in os.listdir(inputfolder):
        burstplot(inputfolder+'/'+filename, outputfolder+'/')
