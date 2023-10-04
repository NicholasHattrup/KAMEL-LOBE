'''
S. Arman Ghaffarizadeh & Gerald J. Wang
Getting over the Hump with KAMEL-LOBE: Kernel-Averaging Method to Eliminate Length-Of-Bin Effects in Radial Distribution Functions
Journal of Chemical Physics (2023)
'''
import numpy as np
from scipy.stats import norm
import scipy.sparse as sp


def KAMEL_LOBE(r,RDF,w=0.015):
    """
    Args:
        r (array): vector of equispaced radii at which RDF is evaluated
        RDF (array): vector of corresponding RDF values
        varargin (float, optional): width of Gaussian kernel (set to 0.015 by default)

    Returns:
        r_tilde (array): vector of equispaced radii at which KAMEL-LOBE RDF is evaluated
        gr_tilde (array): vector of corresponding KAMEL-LOBE RDF values
    """

    Nbins = RDF.shape[0] # number of bins
    delr = r[1]-r[0] # bin width, MATLAB version uses r[2]-r[1]
    m_KL = np.ceil(2*w/delr).astype(int) # number of bins to average over

    if m_KL > 1:

        # Computing T1
        M1 = np.tril(2*np.ones((Nbins,Nbins)), -1)
        M1[np.diag_indices_from(M1)] = 1
        M1[:,0] = 1
        M1[0,:] = 0
        M2 = (np.arange(0, Nbins) ** 2) * delr ** 2
        T1 = M1
        T1 *= M2
   

        # Computing T2
        # Exploits sparsity of T2
        k_KL = 2*m_KL-1
        A1_block = sp.identity(m_KL, format='csr')
        A2_block = sp.lil_matrix((m_KL, Nbins-m_KL))
        fractions = np.zeros((1,k_KL))
        fractions[0,m_KL-1:] = norm.cdf(((np.arange(0,m_KL)+0.5)*delr),0,w)-norm.cdf(((np.arange(0,m_KL)-0.5)*delr),0,w)        
        fractions[0,:m_KL-1] = np.flip(fractions[0,m_KL:2*m_KL-1])
        fractions[0, :] *= 1/np.sum(fractions)
        B_block = sp.diags(np.tile(fractions, (Nbins-2*m_KL, 1)).T, np.arange(0, 2*(m_KL-1)+1), shape=(Nbins-2*m_KL, Nbins))
        T2 = sp.vstack((sp.hstack((A1_block, A2_block)), B_block, sp.hstack((A2_block, A1_block))))
 
        
        # Computing T3
        sq_r = np.zeros((Nbins,1))
        sq_r[1:, 0] = (np.arange(1,Nbins)*delr)**-2
        # Original MATLAB code directly translated to Python
        #K = (-1.)**(np.arange(1,Nbins+1)+1)
        # Python code with possible optimizations 
        K = np.tile([1., -1.],Nbins // 2 + 1)[:Nbins]

        K_sq = np.outer(K, K) # More efficient than tiling 2D array and then element wise multiplication

        KTr = 2*np.tril(K_sq)
        np.fill_diagonal(KTr, 1.) 
        T3 = sq_r*KTr
     


        # Computing gr_tilde
        gr_convert = T3@(T2@(T1@RDF)) # Explicit operation order to reduce redundant matrix multiplications
        gr_tilde = gr_convert[:-2*m_KL]    
        r_tilde = r[:-2*m_KL]
        return r_tilde, gr_tilde
    
    else:
        gr_tilde = RDF
        r_tilde = r
        print('w <= delr/2, no averaging is performed')
        return r_tilde, gr_tilde

