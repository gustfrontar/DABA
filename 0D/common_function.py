#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:45:01 2023

@author: jruiz
"""
import numpy as np
from scipy.linalg import sqrtm

def get_temp_steps( NTemp , Alpha ) :

   #NTemp is the number of tempering steps to be performed.
   #Alpha is a slope coefficient. Larger alpha means only a small part of the information
   #will be assimilated in the first step (and the largest part will be assimilated in the last step).

   dt=1.0/float(NTemp+1)
   steps = np.exp( 1.0 * Alpha / np.arange( dt , 1.0-dt/100.0 , dt ) )
   steps = steps / np.sum(steps)

   return steps


def calc_ref( qr )  :
    #Compute radar reflectivity based on grauple concentration.
        
    nor=8.0e6        #[m^-4]
    ror=1000.0e0     #[Kg/m3]
    pip=np.power( np.pi , 1.75 )   #factor
    cf =1.0e18 * 720 #factor 
    ro=1.0e0

    if qr <= 1.0e-10  :
      qr = 1.0e-10
        
    zr= cf * ( np.power( ro * qr , 1.75 ) )
    zr= zr / ( pip * ( np.power( nor , 0.75 ) ) * ( np.power( ror , 1.75 ) ) )
    
    zr = 10.0*np.log10( zr )
    
    #zr = 10.0e4 * qr + 2.0e-5  #Linear test for debug only. 
    
    return zr

def calc_ref_bis( x ) :

    c = 1221.2518127994958
    if x.size > 1 :
       ref = np.zeros( x.size )
       refmask = x >= 5.0
       ref[ ~refmask ] = 0.0
       ref[ refmask  ] = 10.0 * np.log10( c * ( x[ refmask ]-5.0)**1.75 + 1.0 )
    else :
        if x < 5.0 :
            ref = 0.0
        else :
            ref = 10.0 * np.log10( c * ( x - 5.0 )**1.75 + 1.0 )
    return ref 

def calc_stoc_filter_update( dbz_ens , qr_ens , dbz_obs , R_dbz ) :
    
    dbz_var = np.var( dbz_ens )
    dbz_mean = np.mean( dbz_ens )
    cov=np.cov( np.stack((qr_ens,dbz_ens), axis=0) )[0,1]

    K = ( cov / ( dbz_var + R_dbz ) ) 
    
    
    dbz_a_ens = np.zeros( dbz_ens.shape )
    qr_a_ens  = np.zeros( qr_ens.shape  )

    qr_a_ens = qr_ens + K * ( dbz_obs - dbz_mean + (R_dbz**0.5)*np.random.randn( dbz_ens.size ) ) 

    qr_a_ens[ qr_a_ens  < 0.0] = 0.0

    for ii in range( dbz_ens.size ):
       dbz_a_ens[ii] = calc_ref( qr_a_ens[ii] )
       

    return dbz_a_ens , qr_a_ens


def calc_etkf_filter_update( dbz_ens , qr_ens , dbz_obs , R_dbz ) :
    
    qr_ens_mean = np.mean( qr_ens )
    qr_ens_pert = qr_ens - qr_ens_mean
    
    y_ens = np.zeros( qr_ens.size )
    for ii in range( qr_ens.size ) :
        y_ens[ii] = calc_ref_bis( qr_ens[ii] )
    y_mean = np.mean( y_ens )
    y_pert = y_ens - y_mean
    
    dy = dbz_obs - calc_ref_bis( qr_ens_mean )
    Rinv = 1/R_dbz
    
    Pahat =np.linalg.inv(   np.outer( y_pert.T , Rinv*y_pert ) + ( qr_ens.size -1)*np.identity(qr_ens.size) )
    
    wabar = np.dot( Pahat , y_pert * Rinv * dy )
    Wa = np.real( sqrtm( (qr_ens.size-1)*Pahat) )
    
    qr_a_ens_mean = qr_ens_mean + np.dot( qr_ens_pert , wabar )
    qr_a_ens_pert = np.dot( qr_ens_pert , Wa )
    qr_a_ens = qr_a_ens_mean + qr_a_ens_pert
    dbz_a_ens = np.zeros( qr_a_ens.shape )
    
    qr_a_ens[qr_a_ens < 0.0]=0.0
    for ii in range( qr_ens.size ) :
        dbz_a_ens[ii] = calc_ref_bis( qr_a_ens[ii] )

    
    return dbz_a_ens , qr_a_ens 



def da_statistics( ens_size , qr_mean , qr_std , dbz_tr , qr_model_bias , R_obs , alpha , sample_size )  :
    
    result = dict()
    result['anal_dbz_error'] = np.zeros( sample_size )
    result['gues_dbz_error'] = np.zeros( sample_size )

    result['anal_qr_error'] = np.zeros( sample_size )
    result['gues_qr_error'] = np.zeros( sample_size )

    result['anal_dbz_sprd'] = np.zeros( sample_size )
    result['gues_dbz_sprd'] = np.zeros( sample_size )

    result['anal_qr_sprd'] = np.zeros( sample_size )
    result['gues_qr_sprd'] = np.zeros( sample_size )
    
    result['zero_percentaje_guess'] = np.zeros( sample_size )

    #--------------------
    #Define qr ensemble
    for ir in range( sample_size )   :
        
       qr_ens = qr_mean+qr_std*np.random.randn( ens_size )

       qr_ens[ qr_ens < 0.0 ] = 0.0

       #Define the true and the observations

       qr_true = qr_mean+qr_std*np.random.randn( 1 ) + qr_model_bias
       if qr_true < 0.0 :
           qr_true = 0.0

       dbz_true = calc_ref( qr_true )
       dbz_obs = dbz_true + (R_obs**0.5) * np.random.randn( 1 )
       if dbz_obs < dbz_tr :
          dbz_obs = dbz_tr

       #First compute the ensemble in terms of graupel concentration.
       dbz_ens = np.zeros( np.shape( qr_ens ) )   
       #For each graupel concentration compute the corresponding reflectivity using 
       #the non-linear observation operator.
       for ii in range( ens_size ):
          dbz_ens[ii] = calc_ref( qr_ens[ii] )

       #Truncate the computed reflectivity values so there are no value below 
       #the selected threshold.
       dbz_ens[ dbz_ens < dbz_tr ] = dbz_tr

       dbz_a_ens = dbz_ens
       qr_a_ens  = qr_ens
       for it in range( alpha.size ) :
          dbz_a_ens , qr_a_ens = calc_etkf_filter_update( dbz_a_ens , qr_a_ens , dbz_obs , R_obs * alpha[it] )
       
       dbz_a_ens[ dbz_a_ens < dbz_tr ] = dbz_tr
       qr_a_ens[ qr_a_ens < 0.0 ] = 0.0
      
       result['gues_dbz_error'][ir] = np.mean( dbz_ens )   - dbz_true
       result['anal_dbz_error'][ir] = np.mean( dbz_a_ens ) - dbz_true          

       result['gues_qr_error'][ir] = np.mean( qr_ens )   - qr_true
       result['anal_qr_error'][ir] = np.mean( qr_a_ens ) - qr_true          

       result['gues_dbz_sprd'][ir] = np.std( dbz_ens )   
       result['anal_dbz_sprd'][ir] = np.std( dbz_a_ens )           

       result['gues_qr_sprd'][ir] = np.std( qr_ens )   
       result['anal_qr_sprd'][ir] = np.std( qr_a_ens )     
       result['zero_percentaje_guess'][ir] = np.sum( qr_ens == 0.0 )/ens_size
    
    
    return result
        

def outlier_rmse_filter( rmse ) :
    
    nx , ny = rmse.shape
    for ii in range(1,nx-1) :
        for jj in range(1,ny-1):

            count = 0
            for iii in range(ii-1,ii+1):
                for jjj in range(jj-1,jj+1):
                    if rmse[iii,jjj] > rmse[ii,jj] * 3.0 :
                        count = count + 1
                    if np.isnan( rmse[iii,jjj] ) :
                        count = count + 1
            if count >= 3 :
                rmse[ii,jj] = np.nan
                
    return rmse



    



