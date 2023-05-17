import os
import subprocess
import shlex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from functools import partial
from astropy.utils.console import ProgressBar
from astropy.io import fits
from astropy.table import Table
import json
from multiprocessing import Pool

import seaborn as sns
import scipy.stats as st


import heasoftpy as hsp
import xspec as xs



def _sort_extra_pars(obsids, **kwargs):
    """sort out kwargs, are they a single value (same for all obsid)
    or multi-valued (one per obsid)
        
    """
    nobs = len(obsids)
    extra_pars = [{} for _ in range(nobs)]
    for k,v in kwargs.items():
        try:
            nv = len(v)
        except:
            nv = 0
        if nv == 0:
            for io in range(nobs):
                extra_pars[io][k] = v
        elif nv != nobs:
            raise ValueError(f'Input {k} does not match length of obsids')
        else:
            for io in range(nobs):
                extra_pars[io][k] = v[io]
    return extra_pars


def process_nicer_data_runner(args):
    """Run a single task"""

    obsid, extra_pars = args
    print(f'processing {obsid} ...')

    # input
    inPars = {
        'indir': obsid,
        'geomag_path': 'geomag',
        'filtcolumns': 'NICERV4,3C50',
        'detlist': 'launch,-14,-34',
        'min_fpm': 50,
        'clobber': True, 
        'noprompt': True,
    }
    inPars.update(extra_pars)

    # set local pfile
    pfiles = os.environ['PFILES']
    pdir = hsp.utils.local_pfiles(f'/tmp/{obsid}.pfiles')

    out = hsp.nicerl2(**inPars)
    os.environ['PFILES'] = pfiles
    os.system(f'rm -rf {pdir}')

    if out.returncode == 0: 
        os.system(f'rm -rf {pdir}')
        print(f'{obsid} processed sucessfully!')
    else:
        logfile = f'process_nicer_{obsid}.log'
        print(f'ERROR processing {obsid}; Writing log to {logfile}')
        with open(logfile, 'w') as fp:
            fp.write(out.__str__())

            
def process_nicer_data(obsids, nproc=1, fresh=False, **kwargs):
    """Process a single or multiple nicer obsid
        
    Parameters:
    -----------
        obsid: observation id assumed in current location.
            Either a single obsid or a list. If a list, they are
            run in parallel.
        nproc: maximum number of parallel processess; default: 5
        fresh: fresh data reduction; default: False 
    
    Keywords:
    ---------
        keywords to override the default of nicerl2
    
    """
    
    if isinstance(obsids, str):
        obsids = [obsids]
        
    # sort out kwargs, are they a single value (same for all obsid)
    # or multi-valued (one per obsid)
    extra_pars = _sort_extra_pars(obsids, **kwargs)
    
    unp_obsids, unp_extra_pars = [], []
    for io,oid in enumerate(obsids):
        evtfile = f'{oid}/xti/event_cl/ni{oid}_0mpu7_cl.evt'
        if fresh and os.path.exists(evtfile):
            os.system(f'rm -rf {evtfile}')
        if not os.path.exists(evtfile):
            unp_obsids.append(oid)
            unp_extra_pars.append(extra_pars[io])
    
    obsids = unp_obsids
    extra_pars = unp_extra_pars
    nobs = len(obsids)
    assert(len(extra_pars) == nobs)
    
    
    
    
    with Pool(nproc) as p:
        p.map(process_nicer_data_runner, zip(obsids, extra_pars))
    
    
def extract_nicer_spec_runner(args):
    """Run a single task"""

    obsid, ispec, extra_pars = args
    print(f'processing {obsid} ...')
    
    # set local pfile
    pfiles = os.environ['PFILES']
    pdir = hsp.utils.local_pfiles(f'/tmp/{obsid}.pfiles')
    

    phafile = extra_pars.get('phafile', f'{obsid}/spec/spec{ispec}.pha')
    pharoot = phafile[:-4]
    outdir  = os.path.dirname(phafile)
    os.system(f'mkdir -p {outdir}')

    # input
    inPars = {
        'indir'        : obsid,
        'phafile'      : phafile,
        'rmffile'      : f'{pharoot}.rmf',
        'arffile'      : f'{pharoot}.arf',
        'grouptype'    : 'NONE',
        'loadfile'     : f'{pharoot}.xcm',
        'bkgformat'    : 'file',
        'clobber'      : True,
        'noprompt'     : True,
    }
    
    bgd = extra_pars.get('bgd', ['sc', '3c', 'sw'])
    if 'bgd' in extra_pars: extra_pars.pop('bgd')
    
    inPars.update(extra_pars)
    phafile = inPars['phafile']
    
    
    bg_models = {
        'sc': 'scorpeon',
        '3c': '3c50',
        'sw': 'sw'
    }
    
    for k in list(bg_models.keys()):
        if k not in bgd:
            bg_models.pop(k)

    if len(glob.glob(f'{phafile[:-4]}_??.???')) != 6:

        # get the spectrum and 3 background models
        incr = 'no'
        for blabel,bname in bg_models.items():

            bgfile = f'{phafile[:-4]}_{blabel}.bgd'
            out = hsp.nicerl3_spect(bkgmodeltype=bname, 
                                    bkgfile=bgfile,
                                    incremental=incr,
                                    **inPars)
            if out.returncode == 0: 
                print(f'{obsid}:{blabel} spectra extracted sucessfully!')
                incr = 'yes'

                with fits.open(phafile) as fp:
                    fp['spectrum'].header['backfile'] = bgfile.split('/')[-1]
                    fp['spectrum'].header['respfile'] = inPars['rmffile'].split('/')[-1]
                    fp['spectrum'].header['arffile']  = inPars['arffile'].split('/')[-1]
                    fp.writeto(f'{phafile[:-4]}_{blabel}.pha', overwrite=True)

            else:
                logfile = f'extract_nicer_spec_{obsid}_{blabel}.log'
                print(f'ERROR provessing {obsid}; Writing log to {logfile}')
                with open(logfile, 'w') as fp:
                    fp.write(out.__str__())
        # clean
        os.system(f'rm {outdir}/*bkg_ngt.pi {outdir}/*xcm >dev/null 2>&1')
        os.environ['PFILES'] = pfiles
        os.system(f'rm -rf {pdir}')

            
def extract_nicer_spec(obsids, nproc=1, fresh=False, **kwargs):
    """Extract spectra from a single or multiple nicer obsid.
    Assume we are in folder containing obsids.
    Also assume that heasoft commonds can be accessed.
    nicerl2 is assumed to have been run
    Spectra written to {obsid}/spec
    
        
    Parameters:
    ----------
        obsid: observation id assumed in current location.
            Either a single obsid (obsid, or '{obsid}:{iobs}' so the
            output is spec_{iobs}_sc|3c|sw.) or a list of obsids. 
            If a list, they are run in parallel
        nproc: maximum number of parallel processess; default: 5
    
    Keywords:
    ---------
        keywords to override the default of nicerl3_spect
    
    """
    
    if isinstance(obsids, str):
        obsids = [obsids]
    
    # extract suffic if present or set it as a counter from 1..nobs
    ispec, obsids2 = [], []
    for io,o in enumerate(obsids):
        so = o.split(':')
        obsids2.append(so[0])
        ii = so[1] if len(so) == 2 else f'_{io+1}'
        ispec.append(ii)
    obsids = obsids2
    nobs = len(obsids)
    
    # sort out kwargs, are they a single value (same for all obsid)
    # or multi-valued (one per obsid)
    extra_pars = _sort_extra_pars(obsids, **kwargs)
    
    assert(len(ispec) == nobs == len(extra_pars))
    
                
    with Pool(nproc) as p:
        p.map(extract_nicer_spec_runner, zip(obsids, ispec, extra_pars))


def extract_nicer_lc_runner(args):
    """Run a single task"""

    obsid, extra_pars = args
    print(f'processing {obsid} ...')
    
    # set local pfile
    pfiles = os.environ['PFILES']
    pdir = hsp.utils.local_pfiles(f'/tmp/{obsid}.pfiles')
    
    
    outdir = f'{obsid}/lc'
    os.system(f'mkdir -p {outdir}')
    

    # input
    inPars = {
        'indir'        : obsid,
        'pirange'      : '30:500',
        'timebin'      : 10,
        'bkgformat'    : 'file',
        'incremental'  : 'no',
        'clobber'      : 'yes',
        'noprompt'     : True,
    }
    inPars.update(extra_pars)
    
    bg_models = {
        # 'scorpeon': 'sc',
        # '3c50'    : '3c',
        'sw'      : 'sw'
    }


    if len(glob.glob(f'{outdir}/lc_*.fits')) != 2:


        # get the lightcurve and sw background 
        for bname, blabel in bg_models.items():

            out = hsp.nicerl3_lc(bkgmodeltype=bname, 
                                 lcfile=f'{outdir}/lc_sr_{blabel}.fits',
                                 bkgfile=f'{outdir}/lc_{blabel}.fits',
                                 **inPars)
            if out.returncode == 0: 
                print(f'{obsid}:{blabel} lightcurve extracted sucessfully!')


            else:
                logfile = f'extract_nicer_lc_{obsid}_{blabel}.log'
                print(f'ERROR provessing {obsid}; Writing log to {logfile}')
                with open(logfile, 'w') as fp:
                    fp.write(out.__str__())

        # clean
        os.system(f'rm {outdir}/*.tco')
        os.environ['PFILES'] = pfiles
        os.system(f'rm -rf {pdir}')

            
def extract_nicer_lc(obsids, nproc=1, fresh=False, **kwargs):
    """Extract light curves from a single or multiple nicer obsid.
    Assume we are in folder containing obsids.
    Also assume that heasoft commonds can be accessed.
    nicerl2 is assumed to have been run
    Spectra written to {obsid}/spec
    
        
    Parameters:
    ----------
        obsid: observation id assumed in current location.
            Either a single obsid or a list of obsids. 
            If a list, they are run in parallel
        nproc: maximum number of parallel processess; default: 5
    
    Keywords:
    ---------
        keywords to override the default of nicerl3_lc
    
    """
    
    if isinstance(obsids, str):
        obsids = [obsids]
    nobs = len(obsids)
    
    # sort out kwargs, are they a single value (same for all obsid)
    # or multi-valued (one per obsid)
    extra_pars = _sort_extra_pars(obsids, **kwargs)
    
    assert(nobs == len(extra_pars))
    
                
    with Pool(nproc) as p:
        p.map(extract_nicer_lc_runner, zip(obsids, extra_pars))      
        
        

           
    

def spec_summary(sfiles):
    """Print a short summary of a list of spectra
    
    Parameters:
        sfiles: a list of spectral files
    
    """
    
    # summary of data
    spec_data = []
    fmt = '{:5} | {:12} | {:10.8} | {:10.8} | {:10.3} | {:10.5}\n'
    text = fmt.format('num', 'obsid', 'mjd_s', 'mjd_e', 'rate', 'exposure')
    for ispec,sfile in enumerate(sfiles):
        with fits.open(sfile) as fp:
            obsid = 'NA'
            for k in ['obs_id', 'obsid']:
                if k in fp[1].header:
                    obsid = fp[1].header['obs_id']
                    break
            exposure = fp[1].header['exposure']
            counts   = fp[1].data.field('counts').sum()
            tmid     = np.array([fp[0].header['tstart'], fp[0].header['tstop']])
            mref     = fp[0].header['mjdrefi'] + fp[0].header['mjdreff']
            mjd      = tmid / (24*3600) + mref
            spec_data.append([mjd[0], mjd[1], counts/exposure, exposure/1e3])
            text += fmt.format(ispec+1, obsid, mjd[0], mjd[1], counts/exposure, exposure/1e3)
    print(text)
    spec_data = np.array(spec_data)
    return spec_data, text
