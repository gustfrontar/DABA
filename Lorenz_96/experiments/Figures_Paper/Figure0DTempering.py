# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Feb 26 23:30:13 2019

@author: jruiz

"""
import numpy as np
import matplotlib.pyplot as plt
import common_function as cf
plt.rcParams.update({'font.size': 15})


np.random.seed(70)

#PARAMETERS ---------

ens_size = 5000                      #Size of the ensemble     
kalman_ens_size = 20                 #Size of the ensemble for the ensemble Kalman filter update
x_mean   = 5.0                      #Mean of the original state variable.
x_std    = 2.5                       #Standard deviation of the state variable.
#qr_mean  = 1.0e-4                   #Mean of graupel concentration
#qr_std   = 5.0e-5                   #Standard deviation of graupel concentration.
dbz_tr   = 0.0                       #Minimum dBZ value.
qr_model_bias = 0.0e-4               #When generating the true qr we can add a model bias.                           
R=np.power( 5.0 , 2 )                #Observation error for reflectivity
Ntemp=3                              #Number of tempered steps

alpha = cf.get_temp_steps( Ntemp , 2.0 )

#alpha = np.array([90 , 5,  1])      #Tempering coefficients are computed normalizing the inverse of the components of this vector.
#                                     #the larger the value, the less observations information that is assimilated in each iteration.
#                                     #Usually values are greater for the first iteration an smaller for the rest of them. 

alpha=1 / ( ( 1/alpha ) /  np.sum( 1/alpha ) )

#--------------------
#Define x ensemble
x_ens = x_mean+x_std*np.random.randn( ens_size )

#x_ens[ x_ens < 0.0 ] = 0.0

#Define the true and the observations

x_true = x_mean + 0.8 #qr_std*np.random.randn( 1 ) + qr_model_bias
#if qr_true < 0.0 :
#    qr_true = 0.0

dbz_true = cf.calc_ref_bis( x_true )
dbz_obs = dbz_true + (R**0.5) * np.random.randn( 1 )
if dbz_obs < dbz_tr :
    dbz_obs = dbz_tr

#First compute the ensemble in terms of graupel concentration.
dbz_ens = np.zeros( np.shape( x_ens ) )   
#For each graupel concentration compute the corresponding reflectivity using 
#the non-linear observation operator.
for ii in range( ens_size ):
    dbz_ens[ii] = cf.calc_ref_bis( x_ens[ii] )

#Truncate the computed reflectivity values so there are no value below 
#the selected threshold.
dbz_ens[ dbz_ens < dbz_tr ] = dbz_tr

dbz_a_ens_temp = dbz_ens[0:kalman_ens_size]
x_a_ens_temp  = x_ens[0:kalman_ens_size]

x_ens_temp_evol = np.zeros( ( kalman_ens_size , len(alpha)+1 ) )
x_ens_evol = np.zeros( ( kalman_ens_size , 2 ) )
x_ens_temp_evol[:,0] = x_ens[0:kalman_ens_size]
x_ens_evol[:,0] = x_ens[0:kalman_ens_size]


for it in range( alpha.size ) :
    dbz_a_ens_temp , x_a_ens_temp = cf.calc_etkf_filter_update( dbz_a_ens_temp , x_a_ens_temp , dbz_obs , R * alpha[it] )
    x_ens_temp_evol[:,it+1] = np.copy( x_a_ens_temp )

dbz_a_ens , x_a_ens = cf.calc_etkf_filter_update( dbz_ens[0:kalman_ens_size] , x_ens[0:kalman_ens_size] , dbz_obs , R )
x_ens_evol[:,1] = x_a_ens

pf_w = np.exp(-( dbz_ens - dbz_obs )**2/R  )
pf_w = pf_w / np.sum( pf_w)
 

dbz_a_ens[ dbz_a_ens < dbz_tr ] = dbz_tr
    


fig , axs = plt.subplots( 1 , 3 , figsize=(18,5) , sharey = False )

axs[0].plot( x_ens[0:kalman_ens_size]   , dbz_ens[0:kalman_ens_size] , 'bo' ,markersize=5.0,alpha=0.5)
axs[0].plot( x_a_ens , dbz_a_ens , 'ro' ,markersize=5.0,alpha=0.5)
axs[0].plot( x_true  , dbz_true,'ok',markersize=10.0,label='True')
axs[0].plot( x_true  , dbz_obs,'og',markersize=10.0,label='Obs')
axs[0].set_xlabel('State')
axs[0].set_ylabel('Observation')

axs[0].plot(np.mean( x_ens),np.mean(dbz_ens),'ob',markersize=10.0,label='prior mean')
axs[0].plot(np.mean( x_a_ens),np.mean(dbz_a_ens),'or',markersize=10.0,label='post. mean (ETKF)')
axs[0].plot(np.mean( x_a_ens_temp),np.mean(dbz_a_ens_temp),'om',markersize=10.0,label='post. mean (ETKF-T3)')
axs[0].legend()
axs[0].set_title('(a)')
axs[0].set_xlim(0,25)

#xf_hist , xf_bins = np.histogram( x_ens,bins=np.arange(0.0,25.0,1.0))
#xf_bins_plot= 0.5 * ( xf_bins[0:-1] + xf_bins[1:] )
#axs[1].plot( xf_bins_plot , xf_hist/np.sum(xf_hist) , 'b-' ,label='Prior')
xf_mean = np.mean( x_ens[0:kalman_ens_size] )
xf_std  = np.std( x_ens[0:kalman_ens_size] )
x = np.arange(0,25,0.2)
xf_den= (1.0/(xf_std*np.sqrt(np.pi*2)))*np.exp(-0.5*(x-xf_mean)**2/xf_std**2)
axs[1].plot( x , xf_den , 'b-' ,label='Prior')


xa_mean = np.mean( x_a_ens )
xa_std  = np.std( x_a_ens )
x = np.arange(0,25,0.2)
xa_den= (1.0/(xa_std*np.sqrt(np.pi*2)))*np.exp(-0.5*(x-xa_mean)**2/xa_std**2)
axs[1].plot( x , xa_den , 'r-' ,label='ETKF')

xat_mean = np.mean( x_a_ens_temp )
xat_std  = np.std( x_a_ens_temp )
x = np.arange(0,25,0.2)
xat_den= (1.0/(xat_std*np.sqrt(np.pi*2)))*np.exp(-0.5*(x-xat_mean)**2/xat_std**2)
axs[1].plot( x , xat_den , 'm-' ,label='ETKF-T3')


#xa_hist , xa_bins = np.histogram(x_a_ens,bins=np.arange(0.0,25.0,1.0))
#xa_bins_plot= 0.5 * ( xa_bins[0:-1] + xa_bins[1:] )
#axs[1].plot( xa_bins_plot , xa_hist/np.sum(xa_hist) , 'r-' ,label='Posterior ETKF')

#xat_hist , xat_bins = np.histogram(x_a_ens_temp,bins=np.arange(0.0,25.0,1.0))
#xat_bins_plot= 0.5 * ( xat_bins[0:-1] + xat_bins[1:] )
#axs[1].plot( xat_bins_plot , xat_hist/np.sum(xat_hist) , 'm-' ,label='Posterior ETKF-T3')

xapf_hist , xapf_bins = np.histogram( x_ens , weights=pf_w , bins=np.arange(0.0,25.0,1.0))
xapf_bins_plot= 0.5 * ( xapf_bins[0:-1] + xapf_bins[1:] )
axs[1].plot( xapf_bins_plot , xapf_hist/np.sum(xapf_hist) , 'k--' ,label='Posterior PF')
axs[1].set_xlabel('State')
axs[1].set_ylabel('Density')
axs[1].legend(fontsize=12)
axs[1].set_title('(b)')
axs[1].set_xlim(0,15)

#Select some particles. 

#sel_particles = ( np.random.rand( 10 ) * kalman_ens_size ).astype(int)
pseudo_times = np.arange(Ntemp+1)/Ntemp
#pseudo_times = np.zeros( len(alpha) + 1 )
#pseudo_times[1:] = np.cumsum( (1.0/alpha)/np.sum(1.0/alpha) )
axs[2].plot( x_ens_temp_evol[:,:].T , pseudo_times ,'m-',alpha=0.5)
axs[2].plot( x_ens_evol[:,:].T , np.array([0,1]) ,'r-',alpha=0.5)
axs[2].plot( x_ens_temp_evol[0,:].T , pseudo_times ,'m-',alpha=0.5,label='ETKF-T3')
axs[2].plot( x_ens_evol[0,:].T , np.array([0,1]) ,'r-',alpha=0.5,label='ETKF')
axs[2].set_xlabel('State')
axs[2].set_ylabel('Steps')
axs[2].set_title('(c)')
axs[2].set_xlim(-4,10)
axs[2].legend()

plt.savefig('./Figure0DTempering.png')

# dbzf_hist , dbzf_bins = np.histogram(dbz_ens,bins=np.arange(-30,40,2.5))
# dbzf_bins_plot= 0.5 * ( dbzf_bins[0:-1] + dbzf_bins[1:] )
# ax_y.plot( dbzf_hist/np.sum(dbzf_hist) , dbzf_bins_plot , 'b-' )

# dbza_hist , dbza_bins = np.histogram(dbz_a_ens,bins=np.arange(-30,40,2.5))
# dbza_bins_plot= 0.5 * ( dbza_bins[0:-1] + dbza_bins[1:] )
# ax_y.plot( dbza_hist/np.sum(dbza_hist) , dbza_bins_plot , 'r-' )
