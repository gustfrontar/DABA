import numpy as np

def mean_covar( sample , w = 0 ) :
    if np.size(w) == 1 :
       sample_covar = np.cov( sample , ddof = 0 )
       sample_mean  = np.mean( sample , 1 )
    else           :
       sample_covar = np.cov( sample , aweights=w , ddof = 0 )
       sample_mean  = np.average( sample , axis=1, weights=w ) 

    return sample_mean , sample_covar 

def kld3d(  psample , qhist , q_x , q_y , q_z , q_mean , q_covar , nbins, adjust_mean=True ) :

   tmp_psample = np.copy( psample )
   tmp_qhist   = np.copy( qhist )

   #En este caso q es el posterior estimado con el ensamble grande. Entonces lo ingreso como un histograma 3d precalculado.
   #q_x, q_y y q_z son los limites de dicho histograma. Tambien paso la media y la covarianza de q.

   #El parametro adjust_mean sirve para ajustar la media de ambas distribuciones antes de calcular el kld (si queremos eliminar el efecto del bias
   #y dejar solo el efecto de la covarianza. Incluso tambien podriamos ajustar la varianza para solo quedarnos con las covarianzas.)

   sample_mean = np.mean(tmp_psample , 1 )
   if adjust_mean : #Center the sample of p around the mean of q. 
      tmp_psample[0,:]=tmp_psample[0,:] + q_mean[0]  - sample_mean[0]
      tmp_psample[1,:]=tmp_psample[1,:] + q_mean[1]  - sample_mean[1]
      tmp_psample[2,:]=tmp_psample[2,:] + q_mean[2]  - sample_mean[2]

   xmin=np.min(q_x)
   xmax=np.max(q_x)
   ymin=np.min(q_y)
   ymax=np.max(q_y)
   zmin=np.min(q_z)
   zmax=np.max(q_z)

   [ phist , edges ]= np.histogramdd(tmp_psample.T, bins=(nbins,nbins,nbins),range=((xmin,xmax),(ymin,ymax),(zmin,zmax))  )
   phist = phist / np.sum(phist)
   tmp_qhist = tmp_qhist / np.sum( tmp_qhist )
   
   tmp_qhist[tmp_qhist == 0.0] = np.nan
   phist[phist == 0.0] = np.nan

   #kld = sum(pk * log(pk / qk),  (where pk > 0 and qk > 0 )
   kld = np.nansum( phist * np.log( phist / tmp_qhist ) )

   return kld 

def dist_covar( cov1 , cov2 )  :
   #Distancia entre 2 matrices excluyendo la diagonal.
   #En el caso de matrices de covarianza esto equivale a tomar el coeficiente de correlacion entre cada variable y validar unicamente eso.
   tmp_cov1 = np.copy(cov1)
   tmp_cov2 = np.copy(cov2)

   n=np.shape(cov1)[0]

   for i in range(n)  :
      tmp_cov1[i,:] = tmp_cov1[i,:] / ( tmp_cov1[i,i] ** 0.5 )
      tmp_cov1[:,i] = tmp_cov1[:,i] / ( tmp_cov1[i,i] ** 0.5 ) 
      tmp_cov2[i,:] = tmp_cov2[i,:] / ( tmp_cov2[i,i] ** 0.5 )
      tmp_cov2[:,i] = tmp_cov2[:,i] / ( tmp_cov2[i,i] ** 0.5 )

   dist = np.mean( np.abs( tmp_cov1 - tmp_cov2 ) )

   return dist 


def dist_var( cov1 , cov2 )  :
#Solo nos concentramos en la diagonal para evaluar la varianza.
    dist = np.trace(  np.abs( cov1 - cov2 ) )
    return dist 

def bias_var( cov1 , cov2)  :
    bias = np.trace( cov1 - cov2 )
    return bias

def bias_mean( mean1 , mean2 )  :
    bias = np.sum( mean1 - mean2 )
    return bias
def rmse_mean( mean1 , mean2 )  :
    rmse = np.sqrt( np.mean( (mean1 - mean2)**2 ) )
    return rmse










