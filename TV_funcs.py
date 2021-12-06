from scipy.fft import fftshift, ifftshift, fft2, ifft2 
from scipy.sparse import diags
from scipy.sparse import eye as speye
from numpy.linalg import norm
import numpy as np



def m_transform(m, N, din, k1, k2, order = 'F'):
    min1 = m[:N**2].reshape(N,N,order=order)
    min2 = m[N**2:-1].reshape(N,N,order=order)
    mfout = -1j*(din*k1*fftshift(fft2(min1)) + 
                       din*k2*fftshift(fft2(min2)))
    m = (ifft2(ifftshift(mfout))+m[-1]).real
    return m



def recursive_VBL_TV(XF, y_in, ff, din, k1, k2, gam, lam, return_LS = True, verbose = True):
    y_in = y_in.flatten('F')
    
    N = XF.shape[0]
    Nt = 2*N**2 + 1
        
    vec = np.arange(0,len(XF.flatten('F')),ff)
    M = len(vec)
    print(f'M = {M}')
    
    # LS part
    Xtt = np.zeros((Nt,M), dtype = complex)
    for i in range(M):
        xin = np.zeros(N**2)
        xin[vec[i]] = 1
        xin = xin.reshape(N,N,order='F')
        fxin = din*XF*fftshift(fft2(xin))
        fxin1 = 1j*k1*fxin
        fxin2 = 1j*k2*fxin
        xout1 = ifft2(ifftshift(fxin1))
        xout2 = ifft2(ifftshift(fxin2))
        Xtt[:,i] = np.hstack((xout1.flatten('F'),xout2.flatten('F'),1.))
    Xt = np.copy(Xtt).T.conj()    
    C0 = speye(Nt)
    sub = Xt@C0@Xtt + gam**2*lam*np.eye(M)
    isub = np.linalg.inv(sub)
    K = C0@(Xtt@isub)
    m = K@(y_in.flatten('F')[vec])
    m_ls = np.copy(m)
    
    Cup = np.sum(K*(C0@Xtt),axis=1)
    Cd = (C0.diagonal() - Cup).real/lam
    
    
    # Recursive VBL TV
    mp0 = np.zeros(Nt)
    mp = np.zeros(Nt)    
    M0 = np.copy(M)
    
    j = 1
    while j <= ff - 1:
        vec = np.arange(j-1,len(XF.flatten('F')),ff)
        M = len(vec)
        m0 = np.zeros(Nt)
        XXtt = np.zeros((Nt,M), dtype=complex)
        
        for i in range(M):
            xin = np.zeros(N**2)
            xin[vec[i]] = 1
            xin = xin.reshape(N,N,order='F')
            fxin = din*XF*fftshift(fft2(xin))
            fxin1 = 1j*k1*fxin
            fxin2 = 1j*k2*fxin
            xout1 = ifft2(ifftshift(fxin1))
            xout2 = ifft2(ifftshift(fxin2))
            XXtt[:,i] = np.hstack((xout1.flatten('F'),xout2.flatten('F'),1.))
        XXt = np.copy(XXtt).T.conj()      
        y = y_in.flatten('F')[vec]

        if j == 1:
            Xt = np.copy(XXt)
            Xtt = np.copy(XXtt)
        else:
            Xt = np.vstack((Xt0,XXt))
            Xtt = np.hstack((Xtt0,XXtt))

            y = np.hstack((mp0,y))
            M = len(y)
        k = 1

        while norm(m-m0)/norm(m) > 1e-2 and k <= 2:
            m0 = np.copy(m)
            Cn = np.copy(Cd)
            C0 = diags(np.sqrt(Cd + m**2 + 1e-8))

            sub = Xt@(C0@Xtt) + gam**2*lam*np.eye(M)
            isub = np.linalg.inv(sub)
            K = C0@(Xtt@isub)        

            m = (mp + K@(y-Xt@mp)).real

            Cup = np.sum(K*(C0@Xtt),axis=1)
            Cd = (C0.diagonal() - Cup).real/lam
            k += 1

            if j > 2:
                m0 = np.copy(m)
        mp = np.copy(m)
        j += 1

        if j > 1:
            A = Xt@Xtt
            S,U = np.linalg.eig(A)
            inx = np.argsort(S.real)[::-1]
            Xt0 = U[:,inx[:M0]].T.conj()@Xt
            Xtt0 = Xt0.T.conj()
        mp0 = Xt0@mp
        
        if verbose:
            print(f'iteration = {j-1}')
            
    if return_LS:
        return m_ls, m, Cd
    else:
        return m, Cd
    
    
    
def VBL_TV(XF, y, din, k1, k2, gam, lam, return_LS = True, verbose = True):
    y = y.flatten()
    
    N = XF.shape[0]
    Nt = 2*N**2 + 1
    
    valf, ixf = np.sort(XF.flatten(), kind='stable'), np.argsort(XF.flatten(), kind='stable')
    icritf = len(valf)

    Xtl = np.zeros((N**2,icritf), dtype=complex)
    
    for i in range(icritf):
        fxin = np.zeros(N**2)
        fxin[ixf[-icritf+i]] = valf[-icritf+i]
        fxin = fxin.reshape(N,N)
        xout1 = ifft2(ifftshift(fxin))
        Xtl[:,i] = xout1.flatten()
        
    Xtr = np.zeros((Nt,icritf), dtype = complex)
    tmp1 = 1j*din*k1
    tmp2 = 1j*din*k2
    for i in range(icritf-1):
        fxin1 = np.zeros(N**2, dtype = complex)
        fxin2 = np.zeros(N**2, dtype = complex)
        fxin1[ixf[-icritf+i]] = tmp1.flatten()[ixf[-icritf+i]]       
        fxin2[ixf[-icritf+i]] = tmp2.flatten()[ixf[-icritf+i]] 
        fxin1 = fxin1.reshape(N,N)
        fxin2 = fxin2.reshape(N,N)
        xout1 = N**2*(ifft2(ifftshift(fxin1)))
        xout2 = N**2*(ifft2(ifftshift(fxin2)))
        Xtr[:,i] = np.hstack((xout1.flatten(),xout2.flatten(),0))
    Xtr[:,icritf-1] = np.hstack((np.zeros(2*N**2),N**2))    

    Xl2in = np.linalg.inv((Xtl.T.conj())@Xtl)

    C0 = speye(Nt)
    sub = Xtr.T.conj()@C0@Xtr + gam**2*lam*Xl2in
    isub = np.linalg.inv(sub)
    K = C0@(Xtr@isub)

    m = (K@(Xl2in@(Xtl.T.conj()@y.flatten()))).real
    m_ls = np.copy(m)

    Cup = np.sum(K*(C0@Xtr),axis=1)
    Cd = (C0.diagonal() - Cup).real/lam
    m0 = np.ones((Nt,1))
    k = 1

    while norm(m-m0)/norm(m) > 1e-1:
        m0 = np.copy(m)
        Cn = np.copy(Cd)
        C0 = diags(np.sqrt(m**2 + 1e-8))

        sub = Xtr.T.conj()@C0@Xtr + gam**2*lam*Xl2in
        isub = np.linalg.inv(sub)
        K = C0@(Xtr@isub)
        m = (K@(Xl2in@(Xtl.T.conj()@y.flatten()))).real

        Cup = np.sum(K*(C0@Xtr),axis=1)
        Cd = (C0.diagonal() - Cup).real/lam
        
        if verbose:
            print(f'iteration = {k}')
        k += 1
    
    if return_LS:
        return m_ls, m, Cd
    else:
        return m, Cd
    
    
    
def phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
	"""
	 phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)
	
	Create a Shepp-Logan or modified Shepp-Logan phantom.

	A phantom is a known object (either real or purely mathematical) 
	that is used for testing image reconstruction algorithms.  The 
	Shepp-Logan phantom is a popular mathematical model of a cranial
	slice, made up of a set of ellipses.  This allows rigorous 
	testing of computed tomography (CT) algorithms as it can be 
	analytically transformed with the radon transform (see the 
	function `radon').
	
	Inputs
	------
	n : The edge length of the square image to be produced.
	
	p_type : The type of phantom to produce. Either 
	  "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
	  if `ellipses' is also specified.
	
	ellipses : Custom set of ellipses to use.  These should be in 
	  the form
	  	[[I, a, b, x0, y0, phi],
	  	 [I, a, b, x0, y0, phi],
	  	 ...]
	  where each row defines an ellipse.
	  I : Additive intensity of the ellipse.
	  a : Length of the major axis.
	  b : Length of the minor axis.
	  x0 : Horizontal offset of the centre of the ellipse.
	  y0 : Vertical offset of the centre of the ellipse.
	  phi : Counterclockwise rotation of the ellipse in degrees,
	        measured as the angle between the horizontal axis and 
	        the ellipse major axis.
	  The image bounding box in the algorithm is [-1, -1], [1, 1], 
	  so the values of a, b, x0, y0 should all be specified with
	  respect to this box.
	
	Output
	------
	P : A phantom image.
	
	Usage example
	-------------
	  import matplotlib.pyplot as pl
	  P = phantom ()
	  pl.imshow (P)
	
	References
	----------
	Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue 
	from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
	Feb. 1974, p. 232.
	
	Toft, P.; "The Radon Transform - Theory and Implementation", 
	Ph.D. thesis, Department of Mathematical Modelling, Technical 
	University of Denmark, June 1996.
	
	"""
	
	if (ellipses is None):
		ellipses = _select_phantom (p_type)
	elif (np.size (ellipses, 1) != 6):
		raise AssertionError ("Wrong number of columns in user phantom")
	
	# Blank image
	p = np.zeros ((n, n))

	# Create the pixel grid
	ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]

	for ellip in ellipses:
		I   = ellip [0]
		a2  = ellip [1]**2
		b2  = ellip [2]**2
		x0  = ellip [3]
		y0  = ellip [4]
		phi = ellip [5] * np.pi / 180  # Rotation angle in radians
		
		# Create the offset x and y values for the grid
		x = xgrid - x0
		y = ygrid - y0
		
		cos_p = np.cos (phi) 
		sin_p = np.sin (phi)
		
		# Find the pixels within the ellipse
		locs = (((x * cos_p + y * sin_p)**2) / a2 
              + ((y * cos_p - x * sin_p)**2) / b2) <= 1
		
		# Add the ellipse intensity to those pixels
		p [locs] += I

	return p


def _select_phantom (name):
	if (name.lower () == 'shepp-logan'):
		e = _shepp_logan ()
	elif (name.lower () == 'modified shepp-logan'):
		e = _mod_shepp_logan ()
	else:
		raise ValueError ("Unknown phantom type: %s" % name)
	
	return e


def _shepp_logan ():
	#  Standard head phantom, taken from Shepp & Logan
	return [[   2,   .69,   .92,    0,      0,   0],
	        [-.98, .6624, .8740,    0, -.0184,   0],
	        [-.02, .1100, .3100,  .22,      0, -18],
	        [-.02, .1600, .4100, -.22,      0,  18],
	        [ .01, .2100, .2500,    0,    .35,   0],
	        [ .01, .0460, .0460,    0,     .1,   0],
	        [ .02, .0460, .0460,    0,    -.1,   0],
	        [ .01, .0460, .0230, -.08,  -.605,   0],
	        [ .01, .0230, .0230,    0,  -.606,   0],
	        [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan ():
	#  Modified version of Shepp & Logan's head phantom, 
	#  adjusted to improve contrast.  Taken from Toft.
	return [[   1,   .69,   .92,    0,      0,   0],
	        [-.80, .6624, .8740,    0, -.0184,   0],
	        [-.20, .1100, .3100,  .22,      0, -18],
	        [-.20, .1600, .4100, -.22,      0,  18],
	        [ .10, .2100, .2500,    0,    .35,   0],
	        [ .10, .0460, .0460,    0,     .1,   0],
	        [ .10, .0460, .0460,    0,    -.1,   0],
	        [ .10, .0460, .0230, -.08,  -.605,   0],
	        [ .10, .0230, .0230,    0,  -.606,   0],
	        [ .10, .0230, .0460,  .06,  -.605,   0]]    