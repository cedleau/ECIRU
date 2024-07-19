import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage, misc, stats, signal, interpolate
from scipy.interpolate import CubicSpline

import os, sys, copy, warnings, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import opusFC
import spectrochempy as scp
from spectrochempy import NDDataset
import octavvs.algorithms.atm_correction

from path import operando_path, waterVapour_path

from datetime import datetime

class EC_file3():
    """A class to work on EC files in python.
    This is a wrapper to have useful electrochemistry functions applied to .txt EC-lab export files
    Version 3.0"""
    def __init__(self, path, filename, ref_datetime=None, m_active=None, label=None, verbose=1, decimal=','): 
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
        self.df=pd.read_csv(self.fullpath, sep='\t', skiprows=self.nb_header-1,  encoding='latin', decimal=decimal) 
        
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
 
    def plot_OCV(self, ax=None, x='time/h', alpha=0.3, color='grey', **kwargs):
        """Add colored rectangle when cell is in OCV (ie no current)"""
        for imin, imax in self.LiOCV:
            xmin, xmax = self.df.iloc[imin][x], self.df.iloc[imax][x]
            ax.axvspan(xmin, xmax, alpha=alpha, color=color, **kwargs)
    
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

### Infrared Spectra Time Series
class time_IR_spectra3():
    """A class to import OPUS files in Python"""
    def __init__(self, path=Path('./IR_DATA'), ref_datetime=None, PKA_min=10, overrideOpusDatetime=False):
        AB, ScSm, Sig=[],[],[]
        Ldatet, LPKA, LPRA = [], [], []
        Lfilename = os.listdir(path)
        Lfilename.sort()

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
            self.LdatetOpus = self.Ldatet # Store the time from OPUS in an additional variable
            self.Lfilename = Lfilename

            if overrideOpusDatetime:
                # Set the time of the spectra as last modification time of the file
                # rather than the value recorded by OPUS
                Ldatet=[]
                for filename in Lfilename:
                    filepath  = path / filename
                    ti_m = os.path.getmtime(filepath)                    
                    Ldatet.append(datetime.fromtimestamp(ti_m).replace(tzinfo=ZoneInfo('Europe/Paris')))
                    self.Ldatet = np.array(Ldatet)
            self.overrideOpusDatetime=overrideOpusDatetime

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

    def find_iwn(self, wn):
        return np.argmin(abs(np.array(self.Lwn)-wn))
    
    def find_ith(self, th):
        return np.argmin(abs(np.array(self.Lth)-th))
       

### Vapour

def load_water_vapor_references(path):
    tir = time_IR_spectra3(path=path)
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

tir_dtgs = load_water_vapor_references(path=waterVapour_path)

### Operando 

Operando={}

class OperandoExperimentSimple():
    """A class to automate experiment IR import and processing
- `operando_type`and `num` make up the unique id of the cell. `operando_type` can be anything (like `F1`,`BGSWLK`,`Mai`,...) while `num` must be an integer between 0 and 999. Make sure the folder is formated with three digits : 'F1-112' or 'F1-085' are valid but not 'F1-85')
- `ref_datetime` is an **optional** parameter to set yourself the reference begining time for the experiment. If you put some values, it should be formatted as follow : `datetime.datetime(2023, 8, 22, 9, 56, 52, 544000, tzinfo=datetime.timezone.utc)`"""
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
        Lsubpath = ['ec', 'IR', 'IR_OPERANDO', 'IR_OPERANDO_csv']
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
    
    def load_ec3(self, subpath='ec', verbose=1, decimal=',', **kwargs):
        assert subpath in os.listdir(self.cell_path), f"{subpath} directory cannot be found in {self.cell_path}"
        Lfilename=[filename for filename in os.listdir(self.cell_path/subpath) if '.txt' in filename]
        Lfilename.sort()
        self.Lec = []
        for filename in Lfilename:
            self.Lec.append(EC_file3(path = self.cell_path / 'ec',
                                     filename = filename, ref_datetime = self.ref_datetime, verbose=verbose, decimal=decimal))
            
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
            
    def load_tir3(self, subpath='IR_OPERANDO', override=False, overrideOpusDatetime=False, **kwargs):
        """Load a list of Bruker Opus Files"""
        assert subpath in os.listdir(self.cell_path), f"{subpath} directory cannot be found in {self.cell_path}"
        tir = time_IR_spectra3(self.cell_path / subpath, ref_datetime=self.ref_datetime, overrideOpusDatetime=overrideOpusDatetime, **kwargs)
        if self.tir == None or override:
            self.tir = tir
        return tir

    def load_LIR3(self, subpath='IR', **kwargs):
        """Load a list of Bruker Opus Files"""
        assert subpath in os.listdir(self.cell_path), f"{subpath} directory cannot be found in {self.cell_path}"
        Lfilename = os.listdir(self.cell_path/subpath)
        if len(Lfilename)>0:
            a=scp.read_opus([self.cell_path/subpath/filename for filename in os.listdir(self.cell_path/subpath)])
            setattr(self, subpath, a)

        
    def routine_import(self, verbose=1, decimal_ec=',', overrideOpusDatetime=False):
        """- `decimal_ec` is the decimal number separator for electrochemistry. By default it is a comma for ec-lab files exported on the computers in the lab. Your might use '.' is you exported from your own computer.
- `overrideOpusDatetime`: we have two IR spectrometers in the lab. Somehow, for the new spectro, OPUS software mess up the  time recording when acquiring spectra quickly (every 30 seconds for instance). Setting this parameter to `True` make the code use the file *last modification  date and time* instead of the value inside OPUS file."""
        tir = self.load_tir3(overrideOpusDatetime=overrideOpusDatetime)
        self.load_ec3(verbose=verbose, decimal=decimal_ec)
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

