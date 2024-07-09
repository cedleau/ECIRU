import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os, sys, copy, warnings, time, math, pickle, locale
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize, TABLEAU_COLORS
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import NullFormatter, FixedLocator, AutoMinorLocator
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

from scipy.misc import derivative
from scipy import ndimage, misc, stats, signal, interpolate
from scipy.interpolate import interp1d, PchipInterpolator,CubicSpline

from sklearn.linear_model import LinearRegression


from IPython.display import Markdown, display
from cycler import cycler

import opusFC
import spectrochempy as scp
from spectrochempy import NDDataset
import octavvs.algorithms.atm_correction


%matplotlib inline
#inline/widget
plt.style.use('default')
plt.style.use('seaborn-v0_8-notebook') # 'seaborn-paper'default
rcParams["savefig.bbox"] = 'tight'
rcParams["savefig.dpi"] = '300'
rcParams["axes.labelsize"] = '12'
rcParams["legend.fontsize"] = '12'
rcParams["xtick.labelsize"] = '10'
rcParams["ytick.labelsize"] = '10'


Ewelabel='E(V) vs Li/Li+'
# Path to the directory where can be found the operando cell data
operando_path = Path('../F1-CELLS-DATA')

OperandoSummary=pd.read_excel(operando_path/'Operando_Summary.xlsx')
OperandoSummary.set_index('Name', inplace=True)

class EC_file3():
    """A class to work on EC files in python.
    This is a wrapper to have useful electrochemistry functions applied to .txt EC-lab export files
    Version 3.0"""
    def __init__(self, path, filename, ref_datetime=None, m_active=None, label=None, verbose=1): 
        self.label=label
        self.filename=filename
        self.fullpath = path / filename        
        line2=pd.read_csv(self.fullpath, sep=' : ', skiprows=1, nrows=1, encoding='latin', names=['a','nb'], engine='python')
        self.nb_header=line2['nb'][0]
        with open(self.fullpath) as input_file:
            self.header = [next(input_file) for _ in range(self.nb_header)]
        
        #Parse Characteristic mass from file header
        if m_active==None:
            LcharaMass=[elt for elt in self.header if ('Characteristic mass' in elt) ]
            try:
                s=LcharaMass[0]
                Lmass = s.strip('\n').split(':')[-1].split(' ')[1:]
                m_active = float(Lmass[0].replace(',', '.'))
                if Lmass[1]=='mg':
                    m_active*=1e-3
                elif Lmass[1]=='g':
                    m_active*=1e0
                elif Lmass[1]=='µg':
                    m_active*=1e-6
                else:
                    warnings.warn("Did not recognized unit for characteristic mass") 
            except:
                m_active=1
                warnings.warn("Could not parse characteristic mass from EC-lab file header")
        self.m_active=m_active
        
        #Parse dataset to DataFrame
        self.df=pd.read_csv(self.fullpath, sep='\t', skiprows=self.nb_header-1,  encoding='latin', decimal=',') 
        
        # Detect Positive/Negative Currents/OCV
        if '<I>/mA' in self.df.columns: self.df['I/mA']=self.df['<I>/mA']
        self.df['noCurrent'] = self.df['I/mA']==0
        self.df['Charge'] = self.df['I/mA']>0
        self.df['Discharge'] = self.df['I/mA']<0

        # Process time data : read text time data such as '06/29/2023 17:50:35.4890' and parse it to a Python datetime object
        # Time Delta is then computed from reference datetime, and converted to seconds and hours.
        self.df['datetime']= self.df.loc[:,'time/s'].map(lambda s: datetime.strptime(s, '%m/%d/%Y %H:%M:%S.%f').replace(tzinfo=ZoneInfo('Europe/Paris')))
        if ref_datetime==None:
            ref_datetime=self.df.loc[0, 'datetime']
        self.ref_datetime=ref_datetime
        self.df['time/s'] = self.df['datetime'].map(lambda x:(x-self.ref_datetime).days*86400+(x-self.ref_datetime).seconds)
        self.df['time/h'] = self.df['time/s']/3600
        
        #self.Ldatet = list(map(lambda s: datetime.strptime(s, '%m/%d/%Y %H:%M:%S.%f').replace(tzinfo=ZoneInfo('Europe/Paris')), list(self.df['time/s'])))
        #if ref_datetime==None:
        #    ref_datetime=self.Ldatet[0]
        #self.ref_datetime=ref_datetime
        # Calculate time (s) and (h) from reference datetime
        #self.df['time/s'] = np.array(list(map(lambda x:(x-ref_datetime).days*86400+(x-ref_datetime).seconds, self.Ldatet)))            
        #self.df['time/h']=self.df['time/s']/3600
        
        # Creates (Qo-Q) variable which is easier to look at than (Q-Qo) when looking at anode material
        if '(Q-Qo)/mA.h' in self.df.columns:
            self.df['(Qo-Q)/mA.h']=-self.df['(Q-Qo)/mA.h']
        
        # Manually computes spectific Capacities from capacities and active mass
        for column in ['Capacity/mA.h', 'Q charge/mA.h', 'Q discharge/mA.h','(Q-Qo)/mA.h', '(Qo-Q)/mA.h']:
            if column in self.df.columns:
                self.df[f'{column[:-5]} (mA.h/g)'] = self.df[column]/self.m_active
        self.detect_OCV()
    
    def __repr__(self):
        return ' '.join(['EC-Lab File', str(self.filename)])       

    def detect_OCV(self):
        """A Function to collect indexing of Charging, Discharging and OCV periods"""
        noCurrent_labelled = ndimage.label(self.df.noCurrent.to_numpy().astype(int))
        OCVs = ndimage.find_objects(noCurrent_labelled[0])
        Charge_labelled = ndimage.label(self.df.Charge.to_numpy().astype(int))
        Charges = ndimage.find_objects(Charge_labelled[0])
        Discharge_labelled = ndimage.label(self.df.Discharge.to_numpy().astype(int))
        Discharges = ndimage.find_objects(Discharge_labelled[0])
        self.LiOCV=[]
        self.LiCharges=[]
        self.LiDischarges=[]
        for OCV in OCVs:
            imin, imax= OCV[0].start, OCV[0].stop-1
            self.LiOCV.append((imin, imax))
        for Charge in Charges:
            imin, imax= Charge[0].start, Charge[0].stop-1
            self.LiCharges.append((imin, imax))
        for Discharge in Discharges:
            imin, imax= Discharge[0].start, Discharge[0].stop-1
            self.LiDischarges.append((imin, imax)) 
    
    def plot_header(self):
        for elt in self.header:
            print(elt)
        
    def plot_cycles(self, ax=None, Lcolor=None, k0=None, n=None, x='Capacity/mA.h', y='Ewe/V', labelk0=1):
        if ax==None:
            displayCycles=True
            fig, ax =plt.subplots(1,1)
        if k0==None:
            k0=0
        if n==None:
            n=self.df.half_cycle.unique().max()+1
        if Lcolor==None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            Lcolor = prop_cycle.by_key()['color']       
        for k in range(k0,n):
            df_to_plot=self.df.query(f'half_cycle=={k} & not(noCurrent)')
            if k%2==0:
                label='Cycle '+str(k//2+labelk0)
            else:
                label=None
            ax.plot(df_to_plot[x], df_to_plot[y], color=Lcolor[(k//2)%len(Lcolor)], label=label)

        ax.set_xlabel(x)  
        ax.set_ylabel(y)
        ax.legend()
       
    def plot(self, ax=None, Lcolor=None, k0=None, n=None, x='Capacity/mA.h', y='Ewe/V', scatter=False, label=None, figsize=(6,4), **kwargs):
        if ax==None:
            displayCycles=True
            fig, ax=plt.subplots(1,1, figsize=figsize)
        if label==None:
            label=self.label
        if scatter:
            ax.scatter(self.df[x], self.df[y], label=label, **kwargs)
        else:
            ax.plot(self.df[x], self.df[y], label=label, **kwargs)
        ax.set_xlabel(x)  
        ax.set_ylabel(y)

    def plot_OCV(self, ax=None, x='time/h', alpha=0.3, color='grey', **kwargs):
        """Add colored rectangle when cell is in OCV (ie no current)"""
        for imin, imax in self.LiOCV:
            xmin, xmax = self.df.iloc[imin][x], self.df.iloc[imax][x]
            ax.axvspan(xmin, xmax, alpha=alpha, color=color, **kwargs)
    
    def plotCapacity(self, ax=None, color=None, marker='o', markeredgewidth=1.5, legend=True, label=None, ignoreLast=False, figsize=(5,3), **kwargs):
        """Wrapper for capacity fading plot"""
        plotCapacity(self, ax=ax, color=color, marker=marker, markeredgewidth=markeredgewidth, legend=legend, label=label, figsize=figsize, ignoreLast=ignoreLast, *kwargs)
    
    def compute_dQdV(self, rolling_window=5, median_window=400):
        """
        Apply rolling mean filter to Capacity and Ewe
        Computes dQ/dV
        Finally apply median filter to dQ/dV"""
        self.df['dQ/dV'] = np.gradient(self.df['Capacity/mA.h'].rolling(window=rolling_window).mean(), self.df['Ewe/V'].rolling(window=rolling_window).mean(), edge_order=1)
        self.df['dQ/dV median'] = ndimage.median_filter(self.df['dQ/dV'], size=median_window)

    def dQdV(self, ax=None, Lcolor=None, k0=None, n=None, labelk0=1, rolling_window=5, median_window=400, **kwargs):
        """Apply rolling mean filter to Capacity and Ewe
        Compute dQ/dV
        Finally apply median filter to dQ/dV"""
        if ax==None:
            displayCycles=True
            fig, ax=plt.subplots(1,1)
        if k0==None:
            k0=0
        if n==None:
            n=self.df.half_cycle.unique().max()+1
        if Lcolor==None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            Lcolor = prop_cycle.by_key()['color']
        x='Ewe/V'
        self.df['dQdV']=0
        self.df['dQdV median']=0
        for k in range(k0,n):
            df_to_plot=self.df.query(f'half_cycle=={k} & not(noCurrent)')
            I=df_to_plot.index
            dd=np.gradient(df_to_plot['Capacity/mA.h'].rolling(window=rolling_window).mean(), df_to_plot['Ewe/V'].rolling(window=rolling_window).mean(), edge_order=1)
            self.df.loc[I,'dQdV'] = dd
            self.df.loc[I,'dQdV median']= ndimage.median_filter(self.df['dQdV'][I], size=median_window)
            if k%2==0:
                label='Cycle '+str(k//2+labelk0)
            else:
                label=None
            ax.scatter(self.df[x][I],self.df['dQdV median'][I], color=Lcolor[(k//2)%len(Lcolor)], label=label, **kwargs)
        ax.set_ylim([-6000,6000])
        ax.legend()

    def find_ith(self, th):
        """Find index of a given time (in hour)"""
        return abs(self.df['time/h']-th).idxmin()

def plotCapacity(ec, ax=None, color=None, marker='o', markeredgewidth=1.5, legend=True, label=None, figsize=(5,3), ignoreLast=False, **kwargs):
    """Plot Capacity vs Cycle. To be used along `buildLegendCapacityPlot` function"""
    if ax==None:
        displayCycles=True
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
    label=ec.label if label==None else label  
    
    if color==None:
        color_cycle = ax._get_lines.prop_cycler
        color=next(color_cycle)['color']
    
    # Option to avoid plotting last cycle that is not always complete
    if ignoreLast:
        cycletoIgnore = ec.df['cycle number'].max()
    else:
        cycletoIgnore=-1
    
    m1=(ec.df.query(f'`cycle number`!= {cycletoIgnore}')
        .groupby('cycle number')
        .apply(lambda x: x['Q charge (mA.h/g)'].max())
        .plot(ax=ax, marker=marker, linestyle='', markeredgecolor=color, markerfacecolor=color, markeredgewidth=markeredgewidth, label=label, **kwargs)
        .get_lines()[-1])
    m2=(ec.df.query(f'`cycle number`!= {cycletoIgnore}')
        .groupby('cycle number')
        .apply(lambda x: x['Q discharge (mA.h/g)'].max())
        .plot(ax=ax, marker=marker, linestyle='', markeredgecolor=color, markerfacecolor=(1., 1., 1., 0.), markeredgewidth=markeredgewidth, label=label, **kwargs)
        .get_lines()[-1])
    
    if legend:
        D=ax.scatter(x=[], y=[], marker='$D$', color='k')
        C=ax.scatter(x=[], y=[], marker='$C$', color='k')
        l = ax.legend([(C, D), (m1, m2)], ['',label], numpoints=1,
                        handler_map={tuple: HandlerTuple(ndivide=None)})
    ax.set_ylim(bottom=0)    
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Q Charge / Disch. (mA.h/g)') 

def buildLegendCapacityPlot(ax, Llabel):
    lines=ax.get_lines()
    n=len(lines)//2
    D=ax.scatter(x=[], y=[], marker='$D$', color='k')
    C=ax.scatter(x=[], y=[], marker='$C$', color='k')
    l = ax.legend([(C, D)]+[(lines[2*k], lines[2*k+1]) for k in range(n)], ['']+Llabel, numpoints=1,
                  handler_map={tuple: HandlerTuple(ndivide=None)})

def import_cell(num, celltype='CoinCell', k=0):
    """Simplify import big number of Coincells"""
    L=[]
    if celltype=='CoinCell':
        path=Path('../0_COINCELLS_DATA')
        for filename in os.listdir(path):
            if ('CC-{num:03d}'.format(num=num) in filename) and ('.txt' in filename):
                L.append(EC_file3(path=path, filename=filename, label='CC-{num:03d}'.format(num=num)))
    if celltype=='Swagelok':
        path=Path('../0_SWAGELOCKS_DATA')
        for filename in os.listdir(path):
            if ('SW-{num:03d}'.format(num=num) in filename) and ('.txt' in filename):
                L.append(EC_file3(path=path, filename=filename, label='SW-{num:03d}'.format(num=num)))
    print(L)
    return L[k]

### Infrared Spectra Time Series

class time_IR_spectra3():
    """A class to import OPUS files in Python"""
    def __init__(self, path=Path('./IR_DATA'), ref_datetime=None, PKA_min=10):
        AB, ScSm, Sig=[],[],[]
        Ldatet, LPKA, LPRA = [], [], []
        Lfilename = os.listdir(path)
        Lfilename.sort()
        #for k, filename in enumerate(Lfilename):

        if len(Lfilename)==0:
            warnings.warn(f'Directory {path} is empty - No spectra to load')
            return None
        else:
            for filename in Lfilename:
                filepath = path / filename
                if ('AB', '2D', 'NONE') in opusFC.listContents(filepath):
                    AB_=opusFC.getOpusData(str(filepath), ('AB', '2D', 'NONE'))
                    AB.append(AB_.y)
                    Sig.append(AB_.x)
                    AB_datetime_naive = datetime.strptime(AB_.parameters['DAT']+' '+AB_.parameters['TIM'][:-8], '%d/%m/%Y %H:%M:%S.%f')
                    AB_datetime = AB_datetime_naive.replace(tzinfo=timezone.utc)
                    Ldatet.append(AB_datetime)
                    LPKA.append(AB_.parameters['PKA'])
                    LPRA.append(AB_.parameters['PRA'])
                else:
                    warnings.warn(f'Couldnt absorbance data for {filename}')

                if ('SSC', '2D', 'NONE') in opusFC.listContents(filepath):
                    ScSm_=opusFC.getOpusData(str(filepath), ('SSC', '2D', 'NONE'))
                    ScSm.append(ScSm_.y)

            self.AB, self.ScSm, self.Sig = np.vstack(AB), np.vstack(ScSm), np.vstack(Sig)        
            self.Ldatet, self.LPKA, self.LPRA = np.array(Ldatet), np.array(LPKA), np.array(LPRA)
            self.Lfilename = Lfilename

            if ref_datetime==None:
                ref_datetime = Ldatet[0]
            self.ref_datetime = ref_datetime
            self.Lt = np.array(list(map(lambda x:(x-ref_datetime).days*86400+(x-ref_datetime).seconds, Ldatet)))
            self.Lth = self.Lt/3600
            self.Nt=len(Ldatet)
            self.Lwn = self.Sig[0,:]
            self.Lsigma = self.Lwn
            
            # Additionnaly, Store the Absorbance Data in a Spectrochempy NDDataset
            dwn = scp.Coord(self.Lwn, title="Wavenumbers", history="creation", units=scp.ur.cm**-1)

            dt = scp.Coord(self.Lth, title="Time", history="creation", units='hours', labels=self.Lfilename)
            self.NdAB=scp.NDDataset(self.AB, title='Absorbance')
            self.NdAB.set_coordset(x=dwn, y=dt)
            
            self.IntScSm = 21*np.trapz(self.ScSm, axis=1)
            
            
    def __repr__(self):
        if len(self.LPKA)>0 and len(self.Ldatet)>0:
            return f'Opus time_IR_spectra2 started on {self.Ldatet[0]} with PKA{self.LPKA[0]}'
        else:
            return f'No Spectra Found in {fullpath}'
        
    def substract_spectrum(self, var='AB', iref=0, refSpectrum=None):
        """Older Legacy function
        Compute spectrum variations compared to ref (default is initial spectrum)"""
        Z=getattr(self, var)
        if type(Z)!=type(None):
            if refSpectrum==None:
                refSpectrum=Z[iref, :]
            return Z-np.tile(refSpectrum, (Z.shape[0], 1))
        else:
            warnings.warn("substract_spectrum not executed, no data to apply to")

    def simple_baseline_correction(self, var='AB', lim=[2000, 1850]):
        """Older Legacy function
        Compute spectrum variations compared to ref (default is initial spectrum)"""
        Z=getattr(self, var)
        if type(Z)!=type(None):
            jlim_min, jlim_max=self.find_iwn(lim[0]), self.find_iwn(lim[1])
            return Z-np.tile(Z[:, jlim_min:jlim_max].mean(axis=1), (Z.shape[1],1)).T
        else:
            warnings.warn("simple_baseline_correction not executed, no data to apply to")

    def compute_baseline_variations(self, var='AB', lim=[2000, 1850]):
        """Older Legacy function
        Compute spectrum variations compared to ref (default is initial spectrum)"""
        Z=getattr(self, var)
        if type(Z)!=type(None):
            jlim_min, jlim_max=self.find_iwn(lim[0]), self.find_iwn(lim[1])
            return Z[:, jlim_min:jlim_max].mean(axis=0)
        else:
            warnings.warn("compute_baseline_variations not executed, no data to apply to")
    
    def plot_Wn_vs_t(self, var='AB', x='Lth', y='Lwn', ax=None, cbarlocation='right', cbarlabel='Absorbance', cmap=cm.viridis, I=None, J=None, ylim=[3500,850], **kwargs):
        """to be detailed"""
        if ax==None:
            noAxGiven=True
            fig,ax=plt.subplots(1,1)
        else:
            noAxGiven=False
            fig=ax.get_figure()
        if type(I)==type(None):
            I=range(len(self.Lt))
        if type(J)==type(None):
            J=range(len(self.Lwn))
        
        df=getattr(self, var)
        df2display=df[np.ix_(I, J)]
        
        Lx=getattr(self, x)
        Ly=getattr(self, y)
        
        img=ax.pcolormesh(Lx[I],Ly[J],df2display.T,
                          shading='nearest',cmap=cmap, **kwargs)

        ax.set_ylim(ylim)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Wavenumber ($\mathrm{cm^{-1}}$)')
        if noAxGiven:
            fig.colorbar(img, label=cbarlabel,location=cbarlocation)
        return img

    def plot_PKA(self, ax=None, **kwargs):
        if ax==None:
            noAxGiven=True
            fig,ax=plt.subplots(1,1, figsize=(4,3))
        else:
            noAxGiven=False
            fig=ax.get_figure()
        ax.scatter(self.Lth, self.LPKA, label='PKA', **kwargs)
        ax.scatter(self.Lth, self.LPRA, label='PRA', **kwargs)
        ax.legend()
        
    def plot_ScSm(self, ax=None, mapper=None, **kwargs):
        self.plot_AB(var='ScSm', ax=None, mapper=None, xlim=(6200,800), *kwargs)

    def plot_AB(self, var='AB', ax=None, mapper=None, xlim=(4000,800), **kwargs):
        if ax==None:
            noAxGiven=True
            fig,ax=plt.subplots(1,1, figsize=(4,3))
        else:
            noAxGiven=False
            fig=ax.get_figure()
        
        
        if len(self.Lth)>0:     
            I=list(range(0, len(self.Lth), 5))
            minima = min(self.Lth[I])
            maxima = max(self.Lth[I])

            if mapper == None:
                norm = Normalize(vmin=minima, vmax=maxima, clip=True)
                mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

            jxmin, jxmax = self.find_iwn(xlim[0]), self.find_iwn(xlim[1])
            df=getattr(self, var)        
            if type(df)!=type(None):
                for line, t in zip(df[I,:],self.Lth[I]):
                    ax.plot(self.Lwn[jxmin:jxmax], line[jxmin:jxmax], color=mapper.to_rgba(t), label=None)

                ax.set_xlabel('Wavenumber ($\mathrm{cm^{-1}}$)')
                ax.set_xlim(xlim)
                return mapper
        
    def plot_overview_spectra(self):
        gs_kw = dict(width_ratios=[3, 3, 1], height_ratios=[1])
    
        fig, axs = plt.subplots(1, 3, figsize=(10,3), layout='tight',gridspec_kw=gs_kw)
        ax=axs[0]
        mapper = self.plot_AB(var='ScSm', xlim=(5000, 800), ax=ax)
        ax.set_ylabel('Raw Spectra')
        #ax.set_title(self.cell_id)
        ax=axs[1]
        self.plot_AB(var='AB', xlim=(2000, 800), mapper=mapper, ax=ax)
        ax.set_ylabel('Absorbance')

        ax=axs[2]
        fig.colorbar(mapper, cax=ax, label='Time (h)')
        return fig, ax, mapper
        
    def find_iwn(self, wn):
        return np.argmin(abs(np.array(self.Lwn)-wn))
    
    def find_ith(self, th):
        return np.argmin(abs(np.array(self.Lth)-th))
       

### Vapour

def load_water_vapor_references(path=operando_path):
    tir = time_IR_spectra3(path=operando_path/'Vapour_DTGS'/'Opus')
    tir.ABScSm=-np.log10(tir.ScSm/np.tile(tir.ScSm[0,:].reshape(1,-1), (tir.ScSm.shape[0], 1)))
    Lwnclean=[(6100,5600),(5000, 4050),(3300, 3100),(2600, 2450), (820, 730)]
    Liwnclean=[(tir.find_iwn(pair[0]),tir.find_iwn(pair[1])) for pair in Lwnclean]
    Iclean=[]
    for pairi in Liwnclean:
        Iclean.extend(range(pairi[0], pairi[1]))
    x=tir.Lwn[Iclean]
    y=tir.ABScSm[:, Iclean]
    P=np.polyfit(x, y.T, 2)
    # Correcting baseline of the reference spectra
    tir.ABScSmBaseline=(P[2].reshape(-1,1)*np.power(tir.Sig, 0) + P[1].reshape(-1,1)*np.power(tir.Sig, 1)+ P[0].reshape(-1,1)*np.power(tir.Sig, 2))
    tir.AB = tir.ABScSm - tir.ABScSmBaseline
    tir.NdAB.data=tir.AB

    # These reference spectra are fitted to 3 order polyniolma par morceaux, to compute second derivative, for legacy water correction
    tir.dABScSmCorr=np.zeros(tir.AB.shape)
    tir.ddABScSmCorr=np.zeros(tir.AB.shape)
    for k, spectrum in enumerate(tir.AB):
        x=np.flip(tir.Lsigma)
        y=np.flip(spectrum)
        spl = interpolate.splrep(x,y,k=3) # no smoothing, 3rd order spline
        dy = interpolate.splev(x,spl,der=1) # use those knots to get first derivative
        ddy = interpolate.splev(x,spl,der=2) # use those knots to get second derivative
        tir.dABScSmCorr[k, :] = np.flip(dy)
        tir.ddABScSmCorr[k, :] = np.flip(ddy)
        
    return tir

tir_dtgs = load_water_vapor_references(path='../2023-01_CalibrationInfraredElectrolytes/DTGS')

def correct_water_vapor_2ndDeriv(wn, y, atm, plotResult=False):
    def find_iwn(Lwn, wn):
        return np.argmin(abs(np.array(Lwn)-wn))
        
    # Computing Derivatives of reference spectrum
    dABScSmCorr=np.zeros(atm.shape)
    ddABScSmCorr=np.zeros(atm.shape)
    spectrum = atm
    x_=np.flip(wn)
    y_=np.flip(spectrum)
    spl = interpolate.splrep(x_,y_,k=3) # no smoothing, 3rd order spline
    dy_ = interpolate.splev(x_,spl,der=1) # use those knots to get first derivative
    ddy_ = interpolate.splev(x_,spl,der=2) # use those knots to get second derivative
    dABScSmCorr = np.flip(dy_)
    ddABScSmCorr = np.flip(ddy_)
    
    # Computing Derivatives of y spectrac
    dAB_vap=np.zeros(y.shape)
    ddAB_vap=np.zeros(y.shape)
    for k, spectrum in enumerate(y):
        x_=np.flip(wn)
        y_=np.flip(spectrum)
        spl = interpolate.splrep(x_,y_,k=3) # no smoothing, 3rd order spline
        dy_ = interpolate.splev(x_,spl,der=1) # use those knots to get first derivative
        ddy_ = interpolate.splev(x_,spl,der=2) # use those knots to get second derivative
        dAB_vap[k, :] = np.flip(dy_)
        ddAB_vap[k, :] = np.flip(ddy_)
    
    # Fitting to second derivative of reference spectrum 
    Ivisible=np.array(range(find_iwn(wn, 6000), find_iwn(wn, 800)))
    I3700=np.array(range(find_iwn(wn, 4000), find_iwn(wn, 3400)))
    
    a = ddABScSmCorr[Ivisible].reshape(-1, 1) # Reference spectrum 2nd derivative
    b = ddAB_vap[:, Ivisible].T # 
    c = np.linalg.lstsq(a, b, rcond=None) # x such as b=a@x
    c1=c
    
    a = ddABScSmCorr[I3700].reshape(-1, 1) # Reference spectrum 2nd derivative
    b = ddAB_vap[:, I3700].T
    c = np.linalg.lstsq(a, b, rcond=None) # x such as b=a@x
    c2=c
    
    if plotResult:
        fig, ax = plt.subplots(1, 1, figsize=(5,3), layout='tight')
        ax.scatter(np.arange(y.shape[0]), np.squeeze(c1[0]), label='fit on 6000-800cm-1', s=5)
        ax.scatter(np.arange(y.shape[0]), np.squeeze(c2[0]), label='fit on 4000-3400cm-1', s=5)
        ax.legend()
        ax.set_xlabel('Spectrum number')
        ax.set_ylabel('Water Vapour Contribution for correction')
    
    # Building mask for water peaks regions:
    Lwnfine=[(5600,5100),(4100,3350),(2000, 1300)]
    Liwnfine=[(find_iwn(wn, pair[0]),find_iwn(wn, pair[1])) for pair in Lwnfine]
    Iwnfine=[]
    for pairi in Liwnfine:
        Iwnfine.extend(range(pairi[0], pairi[1]))
        
    # Actually apply the correction
    FineWaterCleaner_month = np.zeros(y.shape)
    FineWaterCleaner_month[:, Iwnfine] = np.transpose(atm[Iwnfine].reshape(-1,1) @  c2[0])
    corrected = y - FineWaterCleaner_month

    #setattr(tir, AB_output, getattr(tir, AB_input) - tir.FineWaterCleaner_month)
    
    # Display the correction
    if plotResult:
        Lwnfine=[(4100,3350),(2000, 1300)]
        fig, axs = plt.subplots(2, 2, figsize=(9,6), layout='tight')
        axs[0, 0].set_ylabel('No Correction')
        axs[1, 0].set_ylabel('Water Vapour Correction')
        for k, xlim, ylim in zip(range(2), Lwnfine, [[-0.025, 0.025], [-0.06, 0.06]]):
            ax=axs[0, k]
            #tir.plot_AB(var=AB_input, ax=ax, xlim=xlim)
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax=axs[1, k]
            #tir.plot_AB(var=AB_output, ax=ax, xlim=xlim)
            ax.set_xlabel('Wavenumber ($\mathregular{cm^{-1}} $)')
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)

    return (corrected,)

### Operando 

Operando={}

class OperandoExperimentSimple():
    """A class to automate experiment IR import and processing"""
    def __init__(self, num:int, operando_type:str = 'F1',  path=operando_path, ref_datetime = None, verbose=1):
        """operando_type:str : String describing the type of experiment
        num:int : number identifier of the cell
        """
        self.num = num
        self.operando_type = operando_type
        # Cell Unique ID, e.g. F1-071
        self.cell_id = f'{operando_type}-{num:03d}'.format(num=num)
        assert self.cell_id in os.listdir(path), f"{self.cell_id} directory cannot be found in given path {path}"
        self.cell_path = path / self.cell_id
        self.ref_datetime = ref_datetime
        Lsubpath = ['ec', 'IR', 'ir_exsitu', 'IR_OPERANDO', 'saved_figures']
        for subpath in Lsubpath:
            assert subpath in os.listdir(self.cell_path), f"{subpath} directory cannot be found in {self.cell_path}"
        
        # Set a reference datetime for operando experiment.
        if self.ref_datetime  == None:
            # Take reference as acquisition of the first operando spectrum
            path_sub = self.cell_path / 'IR_OPERANDO'
            if len(os.listdir(path_sub)) != 0:
                self.ref_datetime = get_opus_datetime((path_sub / os.listdir(path_sub)[0]))
        if self.ref_datetime  == None:
            # Take reference as begining of EC-Lab File
            path_sub = self.cell_path / 'ec'
            Lectxt=[filename for filename in os.listdir(path_sub) if '.txt' in filename]
            if len(Lectxt) != 0:
                self.ref_datetime=EC_file3(filename = Lectxt[0], path=path_sub).ref_datetime
        if self.ref_datetime  == None:
            # Take reference as acquisition of the non-operando bruker opus spectra, with fiber
            path_sub = self.cell_path / 'IR'
            if len(os.listdir(path_sub)) != 0:
                self.ref_datetime = get_opus_datetime((path_sub / os.listdir(path_sub)[0]))
        if self.ref_datetime  == None:
            self.ref_datetime = datetime.fromtimestamp(0, timezone.utc)
        
        self.tir = None
        self.Lec = None
        self.ec = None

    def __repr__(self):
        return f"""Operando Cell {self.cell_id}"""
    
    def load_ec3(self, subpath='ec', verbose=1, **kwargs):
        assert subpath in os.listdir(self.cell_path), f"{subpath} directory cannot be found in {self.cell_path}"
        Lfilename=[filename for filename in os.listdir(self.cell_path/subpath) if '.txt' in filename]
        Lfilename.sort()
        self.Lec = []
        for filename in Lfilename:
            self.Lec.append(EC_file3(path = self.cell_path / 'ec',
                                     filename = filename, ref_datetime = self.ref_datetime, verbose=verbose))
            
        # If they are several EC-lab files, there are connected one after the other, this part of code edits 'accumulated values' so they match to each other
        Lec=self.Lec
        if len(Lec)>1:
            for column in ['x', '(Q-Qo)/mA.h','(Q-Qo) (mA.h/g)']:
                for k in range(1, len(Lec)):
                    ec_0, ec_ = Lec[k-1], Lec[k]
                    if (column in ec_0.df.columns) and (column in ec_.df.columns):
                        #print(ec_0.df[column].iloc[-1], ec_.df[column].iloc[0])
                        ec_.df[column]=ec_.df[column] - ec_.df[column].iloc[0] + ec_0.df[column].iloc[-1]    
        
        # This is where the actual file concatenation takes place
        if len(self.Lec)>0:
            self.ec=copy.deepcopy(self.Lec[0])
            self.ec.df=(pd.concat([ec_.df for ec_ in self.Lec], ignore_index=True)
                        .reset_index(drop=True))
            self.ec.detect_OCV()
            
    def load_tir3(self, subpath='IR_OPERANDO', override = False, **kwargs):
        """Load a list of Bruker Opus Files"""
        assert subpath in os.listdir(self.cell_path), f"{subpath} directory cannot be found in {self.cell_path}"
        tir = time_IR_spectra3(self.cell_path / subpath, ref_datetime=self.ref_datetime, **kwargs)
        if self.tir == None or override:
            self.tir = tir
        return tir

    def load_LIR3(self, subpath='IR', **kwargs):
        """Load a list of Bruker Opus Files"""
        assert subpath in os.listdir(self.cell_path), f"{subpath} directory cannot be found in {self.cell_path}"
        a=scp.read_opus([self.cell_path/subpath/filename for filename in os.listdir(self.cell_path/subpath)])
        
    def routine_import(self, verbose=1):
        self.load_tir3()
        tir = self.tir

        # Interpolating to Lwn vector 
        atm = interpolate_spectra(self.tir.Lwn, tir_dtgs.Lwn, tir_dtgs.NdAB[2, :].data)

        ## Remove initial spectrum, do a simple baseline correction and corrects water vapour
        tir.ABb=tir.substract_spectrum(var='AB')
        tir.ABc=tir.simple_baseline_correction(var='ABb')
        #tir.ABd=correct_water_vapor_2ndDeriv(wn=tir.Lwn, y=tir.ABc, atm=tir_dtgs.NdAB[2, :].data.squeeze())
        tir.ABd=octavvs.algorithms.atm_correction.atmospheric(wn=tir.Lwn, y=tir.ABc, atm=atm)[0]
        
        ## Do a simple baseline correction and corrects water vapour
        tir.ABe=tir.simple_baseline_correction(var='AB')
        #tir.ABf=correct_water_vapor_2ndDeriv(wn=tir.Lwn, y=tir.ABe, atm=tir_dtgs.NdAB[2, :].data.squeeze())
        tir.ABf=octavvs.algorithms.atm_correction.atmospheric(wn=tir.Lwn, y=tir.ABe, atm=atm)[0]

        tir.Lbaseline = tir.compute_baseline_variations()
        self.load_ec3(verbose=verbose)
        self.load_LIR3()
        
        df=pd.DataFrame(tir.NdAB.data, index=tir.NdAB.y.data, columns=tir.NdAB.x.data)
        df.to_csv(self.cell_path/'IR_OPERANDO_csv'/f'{self.cell_id}_raw.csv')
        
def interpolate_spectra(Lwn_new, Lwn, y):
    flip=Lwn[0]>Lwn[-1]
    """Encapsulation of CubicSpline"""
    # Interpolating to Lwn vector 
    if flip:
        xnew = np.flip(Lwn_new.squeeze())
        x = np.flip(Lwn.squeeze())
        y = np.flip(y.squeeze())
        spl = CubicSpline(x, y)
        ynew =  np.flip(spl(xnew))
    else:
        xnew = Lwn_new.squeeze()
        x = Lwn.squeeze()
        y = y.squeeze()
        spl = CubicSpline(x, y)
        ynew =  spl(xnew)
    return ynew
        
def get_opus_datetime(filepath):
    if ('AB', '2D', 'NONE') in opusFC.listContents(filepath):
        AB_=opusFC.getOpusData(str(filepath), ('AB', '2D', 'NONE'))
        AB_datetime_naive = datetime.strptime(AB_.parameters['DAT']+' '+AB_.parameters['TIM'][:-8], '%d/%m/%Y %H:%M:%S.%f')
        AB_datetime = AB_datetime_naive.replace(tzinfo=timezone.utc)
        return AB_datetime
        
def export_tir_to_csv(tir, attr='NdAB_var_SEI', cell_id='Cell-001', Lwn = [1790.,1767.,1180.,1212.,842.,]):
    if not 'export' in os.listdir(): os.mkdir('export')
    ## Wavenumbers to export
    NdAB = getattr(tir, attr)
    assert type(NdAB) == scp.core.dataset.nddataset.NDDataset , 'Make sure the attr you want is NdAB (spectrochempy) and not AB (numpy)'
    df2export = np.array([NdAB[:, wn].data.squeeze() for wn in Lwn]).T
    df2export = pd.DataFrame(df2export, columns=Lwn, index= tir.Lth)
    df2export.to_csv(f'export/{cell_id}_SelectedWavenumerConcentration.csv')

