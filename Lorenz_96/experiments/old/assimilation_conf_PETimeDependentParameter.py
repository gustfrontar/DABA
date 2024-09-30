import numpy as np

#=================================================================
#Parameter estimation Experiment 3
#In this experiment we will estimate only one parameter (forcing mean)
#The true parameter is time dependent following an AR1 process.
#=================================================================

GeneralConf=dict()

GeneralConf['ExpName']='LETKF_PETimeDependentParameter'                 #Experiment name.
GeneralConf['DataPath']='./data/Assimilation/'                          #Data output path
GeneralConf['FigPath']='./figs/Assimilation/'                           #Figures output path
GeneralConf['RunSave']=True                                             #Save the output
GeneralConf['OutFile']='Assimilation' + GeneralConf['ExpName'] + '.npz' #Output file
GeneralConf['RunPlotState']=True                                        #Plot State Estimation Diagnostics
GeneralConf['RunPlotForcing']=False                                     #Plot Forcing Estimation Diagnostics
GeneralConf['RunPlotParameters']=True                                   #Plot Parameter Estimation Diagnostics

#Obs data, obs configuration and nature run configuration are stored
#in this file.
GeneralConf['ObsFile']='./data/Nature/NatureTimeDependentParameter.npz'

#=================================================================
# MODEL SECTION : 
#=================================================================
#General model section

ModelConf=dict()

#General model section

ModelConf['nx'] =  40                                   #Number of large-scale state variables
ModelConf['dt']  =0.0125                                #Time step for large-scale variables (do not change)

#Forcing section

ModelConf['Coef']=np.array([7.75])                     #Coefficient of parametrized forcing (polynom coefficients starting from coef[0]*x^0 + coef[1]*x ... ) 

ModelConf['NCoef']=np.size(ModelConf['Coef'])

#Space dependent parameter

ModelConf['FSpaceDependent']=False                      #If the forcing parameters will depend on the location.
ModelConf['FSpaceAmplitude']=np.array([1])              #Amplitude of space variantions (for each coefficient)
ModelConf['FSpaceFreq']     =1                          #Use integers >= 1

#Parameter random walk          

ModelConf['EnablePRF']=False                            #Activate Parameter random walk
ModelConf['CSigma']=np.array([0])                       #Parameter random walk sigma
ModelConf['CPhi'  ]=1.0                                 #Parameter random walk phi

#State random forcing parameters

ModelConf['EnableSRF']=False                            #Activate State random forcing.
ModelConf['XSigma']=0.0                                 #Amplitude of the random walk
ModelConf['XPhi'  ]=1.0                                 #Time autocorrelation parameter

ModelConf['XLoc'  ]=np.arange(1,ModelConf['nx']+1)      #Location of model grid points (1-nx)

#Two scale model parameters

ModelConf['TwoScaleParameters']=np.array([10,10,0])     #Small scale and coupling parameters C , B and Hint
                                                        #Set Hint /= 0 to enable two scale model
                                              
ModelConf['nxss']= ModelConf['nx'] * 8                  #Number of small scale variables
ModelConf['dtss']= ModelConf['dt'] / 5                  #Time step increment for the small scale variables


#=================================================================
#  DATA ASSIMILATION SECTION :
#=================================================================

DAConf=dict()

DAConf['NEns'] = 30                                  #Number of ensemble members

DAConf['Twin'] = False                               #When True, model configuration will be replaced by the model configuration in the nature run.

DAConf['Freq'] = 4                                   #Assimilation frequency (in number of time steps)
DAConf['TSFreq'] = 4                                 #Intra window ensemble output frequency (for 4D Data assimilation)

DAConf['InfCoefs']=np.array([1.06,0.0,0.0,0.0,0.0])   #Mult inf, RTPS, RTPP, EPES, Additive inflation

DAConf['LocScales']=np.array([4.0,-1.0])             #Localization scale is space and time (negative means no localization)

#Initial state ensemble.
DAConf['InitialXSigma']=1                            #Initial ensemble spread for state variables.

DAConf['UpdateSmoothCoef']=0.0                       #Data assimilation update smooth (for parameter estimation only)

#Parameter estimation/perturbation 

DAConf['InitialPSigma']=np.array([0.035,0,0])         #Initial ensemble spread for the parameters. (0 means no parameter estimation)

DAConf['InfCoefsP']=np.array([1.3,0.0,0.0,0.0,0.0])  #Mult inf, RTPS, RTPP, EPES, Additive inflation

DAConf['UpdateSmoothCoefP']=0.0                      #Data assimilation update smooth (for parameter estimation only)

DAConf['EstimateParameters']=True                    #Wether parameters will be estimated or not.

DAConf['ParameterLocalizationType']=1                #1-Global parameter (no loc), 2-Averaged local estimation , 3-Full local estimation
 
DAConf['LocScalesP']=np.array([4.0,-1.0])            #Localization scale for the parameter: to be used with ParameterLocalizationTypes 2 or 3.


