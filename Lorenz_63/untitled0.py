

# load python modules
import numpy.random as rnd
import numpy as np
# my modules
import pf_gra as gra
#import ass.resample as rspl
import com.com as com
#from com.com import inv_svd as inv
from com.com import inv_svd as inv
#"""
#Different flavours of particle filters
#"""

class FILTER:
    
# La que esta funcionando es la del main_pf.py
    #-----------------------------------------------------
    def __init__(self,Obs,Fmdl,assmtd='sir',resampleRate=0.5,ltdepR=0,resample=None,lgraf=0):
        # as a rule all the (free-tuning) parameters should go here
        #import np.linalg.inv as inv
        
        [nx,n]=sqQ.shape

        self.sqQ=Fmdl.sqQ
        self.sqR=Obs.sqR
        self.H=Obs.H
        
        self.integ=Fmdl.integ
        self.assmtd=assmtd

        
        self.lgraf=lgraf # plot the output
        self.lgordon=0 # 0: resampling with threshold, 1: resampling every ass step (original Gordon algorithm).
        self.ltdepR=ltdepR # time dependent R switch 
        
        if resample is not None:
            self.resample=eval('self.'+resample)
            #eval('rspl.'+resample)
        else:
            self.resample=None
            
        R=np.dot(self.sqR,self.sqR.T)
        Q=np.dot(self.sqQ,self.sqQ.T)
        
        #Qinv=np.linalg.inv(Q)
        self.Rinv=np.linalg.pinv(R)
        self.Qinv=np.linalg.pinv(Q)

        self.resampleRate=resampleRate #number of effective particles required 
#        self.Nthreshold=0.5 #*npa #number of effective particles required 

        return
    
    ##- 1 -------------------------------------------------
    def asscy(self,Xa,yo_t,fmdl,ass):   
        """
         SIR particle filter (Gordon et al 1993)

              Observations, yo_t are expected at it=1 (xa is at it=0)
         """
    #    from com.com import sqrt_svd as sqrtm

        [nx, npa] = Xa.shape
        [ny, ncy] = yo_t.shape
        self.Nthreshold=self.resampleRate*npa
        self.ncy=ncy

        xa_t=np.zeros((nx,npa,ncy))
        xf_t=np.zeros((nx,npa,ncy))
        Mx_t=np.zeros((nx,npa,ncy))
        w_t=np.zeros((npa,ncy))

        w=np.zeros(npa)+1./npa # initial weights (MC sampling)
        print 'sqQ: ',self.sqQ
        
        # do assimilation cycles
        for icy in range(ncy):
            # Evolve particles
            Mx = fmdl.integ(Xa)

            if self.sqQ is not None:
                #  Add model error.  
                wrk = rnd.normal(0, 1, [nx,npa])
                Xf = Mx + np.dot(self.sqQ,wrk)
            else:
                quit('Particle filter requires Model Error')
                
            self.icy=icy
            # plot posterior for last cycle
            #if icy == ncy-1:
            #     self.lgraf=1
            
            # time depentent R
            if self.ltdepR:
                if self.Rinv.size == 1:
                    #print icy,fmdl.R_t[...,icy]
                    self.Rinv=1./fmdl.R_t[...,icy]
                else:
                    self.Rinv=inv(fmdl.R_t[...,icy])

            # asimilacion
            Xa,w = ass(Xf,Mx,yo_t[:,icy],w)

            # save time dependent variables
            xa_t[:,:,icy] = Xa[:,:]
            xf_t[:,:,icy] = Xf[:,:]
            Mx_t[:,:,icy] = Mx[:,:]
            w_t[:,icy] = w[:]

        # end do icy

        return xa_t,xf_t,w_t,Mx_t

    #- 2 -------------------------------------------------
    def sir(self,Xf0,Mx,yo,wold):
        """ 
           Assimilation cycle using the particle filter

        Algorithm 4. SIR PF from Arulampalam et al 2002 
         (originally Gordon et al 1993)
              q=p(x_k|x_{k-1})
              resampling every cycle
        (The inputs are all that are needed for sir and for steinpf
        """

        [nx, npa] = Xf0.shape
        
        Xf=np.zeros([nx,npa])
        Xf[:,:]=Xf0[:,:]
        w=np.zeros(npa)

        for ip in range(1,npa):
            Hxf=np.dot(self.H,Xf[:,ip])
            # xf is already the sample at tk since we assume q(x_k|x_k-1^i)=p(x_k|x_k-1^i)
            w[ip] = self.obslik(yo,Hxf,self.Rinv) # weight(y,hx,Rinv,fac). Assumes resampling every time step.
            #print 'yo-Hx',yo-Hxf
            #print 'ip, w',ip,w[ip]
            
        wnorm=w/np.sum(w)

        if self.resample is not None:
            Neff=np.sum(wnorm**2)**-1

            #print 'Nro de particulas efectivas: ',Neff
            if self.lgordon: # siempre hace resampling
                Neff=0

            if Neff<self.Nthreshold: #threshold --> resampling
                indexres=self.resample(wnorm)
                Xf=Xf[:,indexres]
                wnorm=np.zeros(npa)+1.0/npa

        #TMP
        #if (self.lgraf and (self.icy == 115 or self.icy == 123 or self.icy == 138)):
        #    gra.posterior(yo,Xf0,Xf,Mx,wold,self.Qinv,self.Rinv,self.H)
                
        if (self.lgraf and self.icy == self.ncy-1):
            gra.posterior(yo,Xf0,Xf,Mx,wold,self.Qinv,self.Rinv,self.H)
       

        return Xf,wnorm

    #- 3 -------------------------------------------------
    def obslik(self,y,Hxf,Rinv):
        """
            Compute Gaussian likelihood of observations y, given H(state)=Hx
        """

        nx=Hxf.size
        inc=y-Hxf
        d1=np.dot(Rinv,inc)
        pd=np.exp(-0.5*np.dot(inc.T,d1))

        return pd

    #- 1 -------------------------------------------------

    def systematic(weights):
        """    
        Draw a single random number, u_j= [(j-1)+rdn]/Npa
          Take x_k from multinomial distribution
        Taken from pyfilt 0.1?
        """
        N = len(weights)

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.random() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
    
        cumulative_sum = np.cumsum(weights)

        i, j = 0, 0

        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        return indexes

    #- 5 -------------------------------------------------

    def residual(weights):
        """ Performs the residual resampling algorithm used by particle filters.
        Taken from pyfilt 0.1?
        Parameters:
        ----------
        weights  
        Returns:
        -------
        indexes : ndarray of ints
        array of indexes into the weights defining the resample.
        """

        N = len(weights)
        indexes = np.zeros(N, 'i')
        
        # take int(N*w) copies of each weight, which ensures particles with the
        # same weight are drawn uniformly
        num_copies = (np.floor(N*np.asarray(weights))).astype(int)
        k = 0
        for i in range(N):
            for _ in range(num_copies[i]): # make n copies
                indexes[k] = i
                k += 1
                
        # use multinormal resample on the residual to fill up the rest. 
        residual = weights - num_copies     # get fractional part
        residual /= sum(residual)           # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
        ran1=np.random.uniform(size=N-k)
        indexes[k:N] = np.searchsorted(cumulative_sum, ran1)

        return indexes