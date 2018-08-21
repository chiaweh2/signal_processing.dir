from regression import regress
import numpy as np
import sys


class grace_postprocessing():
    """
    The class is created to include the postprocessing steps needed 
    for the GRACE satellite data.
    
    method :
        1) destripe_ocean (based on Chambers 2006 paper)
    
    """
    
    def __init__(self, cutoff=None):
        """
        Parameters:
            self.cutoff: optional, default = None
                the cutoff is to set the degree of cutoff in the Stokes' 
                coefficents (clms, slms)
            
        """
        
        self.cutoff=cutoff
        

    def destripe_ocean(self, clms, slms) :
        """
        Written by Chia-Wei Hsu
        Based on Chambers 2006

          The signal processing function is designed to applied on a set of Stokes' coeffiences.
          Assuming the clms, slms are stored in the matrix of size = [l,m].
          Lower degree (l) and order (m) (11x11) of the coefficients are unchanged,
          as well as all order 0 and order 1 coefficients.
          A 5th order polynomial is fitted to even or odd degree (n) to the remaining coefficients
          for each order (m) greater than 2 from n = 12 (or n = m if m > 11) up to n = 60.
          Only one polynomial is computed for each odd or even set for a given order.
          Only coefficients up to m = 40 are de-striped. Every coefficient above n = 40, m = 40 is set to zero

        CAUTION!!!! input clm slm must be 61*61  cannot be 41*41
        """


        if clm1.shape[0] == 1 or slm1.shape[0] == 1 :
           sys.exit('please input the correct array')
        lmost=60
        #------------------   program start  ----------------------

        # matrix size declare
        clmeven=np.zeros(lmost,dtype=np.float64)
        slmeven=np.zeros(lmost,dtype=np.float64)
        clmodd=np.zeros(lmost+1,dtype=np.float64)
        slmodd=np.zeros(lmost+1,dtype=np.float64)
        clmsm=np.zeros([lmost+1,lmost+1],dtype=np.float64)
        slmsm=np.zeros([lmost+1,lmost+1],dtype=np.float64)


        # constant setting
        pi=np.arccos(-1.)
        dtr=pi/np.float64(180.)
        lmin=12                     # clm/slm min degree
        lmax=40                     # clm/slm max degree

        for m in range(2,int(lmax+1)) :
            # put the even and odd l's into their own arrays
            ieven=-1
            iodd=-1
            leven=np.zeros(lmost,dtype=np.int)
            lodd=np.zeros(lmost,dtype=np.int)
            for l in range(int(m),int(lmost+1)) :
                if ((np.remainder(l,2) == 0) or (np.remainder(l,2) == 2)) :
                   ieven=ieven+1
                   leven[ieven]=l
                   clmeven[ieven]=clm1[l,m]
                   slmeven[ieven]=slm1[l,m]
                else :
                   iodd=iodd+1
                   lodd[iodd]=l
                   clmodd[iodd]=clm1[l,m]
                   slmodd[iodd]=slm1[l,m]

            # create 5th order polynomial
            leven_temp=leven[np.where(leven>=lmin)]
            leven_index=np.zeros(leven_temp.shape,dtype=np.int)
            #leven_ttemp=leven_temp[np.where(leven_temp<=lmax)]
            #leven_iindex=np.zeros(leven_ttemp.shape,dtype=np.int)
            lodd_temp=lodd[np.where(lodd>=lmin)]
            lodd_index=np.zeros(lodd_temp.shape,dtype=np.int)
            #lodd_ttemp=lodd_temp[np.where(lodd_temp<=lmax)]
            #lodd_iindex=np.zeros(lodd_ttemp.shape,dtype=np.int)

            lll=0
            #llll=0
            for ll in range(leven.shape[0]):
                if leven[ll] in leven_temp:
                   leven_index[lll]=ll
                   lll=lll+1
               # if leven[ll] in leven_ttemp:
               #    leven_iindex[llll]=ll
               #    llll=llll+1
            lll=0
            #llll=0
            for ll in range(lodd.shape[0]):
                if lodd[ll] in lodd_temp:
                   lodd_index[lll]=ll
                   lll=lll+1
               # if lodd[ll] in lodd_ttemp:
               #    lodd_iindex[llll]=ll
               #    llll=llll+1

            #print lodd_index,lodd_temp
            #sys.exit('stop')
            clmeven_temp=clmeven[leven_index]
            slmeven_temp=slmeven[leven_index]
            clmodd_temp=clmodd[lodd_index]
            slmodd_temp=slmodd[lodd_index]

            beta_clmeven=np.array(time_multi_regress(clmeven_temp,leven_temp,flag=8)['beta'])
            beta_slmeven=np.array(time_multi_regress(slmeven_temp,leven_temp,flag=8)['beta'])
            beta_clmodd=np.array(time_multi_regress(clmodd_temp,lodd_temp,flag=8)['beta'])
            beta_slmodd=np.array(time_multi_regress(slmodd_temp,lodd_temp,flag=8)['beta'])

            clmeven_stripe=beta_clmeven[0]+beta_clmeven[1]*leven_temp+beta_clmeven[2]*leven_temp**2+beta_clmeven[3]*leven_temp**3\
                           +beta_clmeven[4]*leven_temp**4+beta_clmeven[5]*leven_temp**5
            slmeven_stripe=beta_slmeven[0]+beta_slmeven[1]*leven_temp+beta_slmeven[2]*leven_temp**2+beta_slmeven[3]*leven_temp**3\
                           +beta_slmeven[4]*leven_temp**4+beta_slmeven[5]*leven_temp**5
            clmodd_stripe=beta_clmodd[0]+beta_clmodd[1]*lodd_temp+beta_clmodd[2]*lodd_temp**2+beta_clmodd[3]*lodd_temp**3\
                           +beta_clmodd[4]*lodd_temp**4+beta_clmodd[5]*lodd_temp**5
            slmodd_stripe=beta_slmodd[0]+beta_slmodd[1]*lodd_temp+beta_slmodd[2]*lodd_temp**2+beta_slmodd[3]*lodd_temp**3\
                           +beta_slmodd[4]*lodd_temp**4+beta_slmodd[5]*lodd_temp**5

            # subtract polynomial from original array
            clmeven_temp=clmeven_temp-clmeven_stripe
            slmeven_temp=slmeven_temp-slmeven_stripe
            clmodd_temp=clmodd_temp-clmodd_stripe
            slmodd_temp=slmodd_temp-slmodd_stripe

            # substitude the destriped solution with the original array
            clmeven[leven_index]=clmeven_temp
            slmeven[leven_index]=slmeven_temp
            clmodd[lodd_index]=clmodd_temp
            slmodd[lodd_index]=slmodd_temp

            # put the even and odd l's back to their own arrays
            ieven=-1
            iodd=-1
            for l in leven :
                ieven=ieven+1
                clm1[l,m]=clmeven[ieven]
                slm1[l,m]=slmeven[ieven]
            for l in lodd :
                iodd=iodd+1
                clm1[l,m]=clmodd[iodd]
                slm1[l,m]=slmodd[iodd]

        return{'clm':clm1,'slm':slm1}
