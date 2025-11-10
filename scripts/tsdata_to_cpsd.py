import os
import pickle
import sine_parameter_utils

import config
import numpy
from scipy.signal import welch, csd, windows


def demean(X, normalize=False):
    '''
    Temporally demean time series data.
    Python port of demean.m
    https://osf.io/hvnjt/overview
    
    Input
    X: ndarray, shape (n_channels, n_timepoints, n_trials), multi-trial time series data.
    normalize: bool, normalize variance to 1 (default: False)
    
    Output
    Y: ndarray, demeaned (optionally normalized) time series
    '''

    n, m, N = X.shape
    Y = X.reshape(n, m*N)
    Y = Y - numpy.mean(Y, axis=1, keepdims=True)
    
    if normalize:
    
        Y = Y / numpy.std(Y, axis=1, keepdims=True)
    return Y.reshape(n, m, N)




def tsdata_to_cpsd(X, fres, method='WELCH', window=None, noverlap=None, nw=3, ntapers=None):
    '''
    Estimate cross-power spectral density from time series data.
    Python port of tsdata_to_cpsd.m
    https://osf.io/hvnjt/overview
    
    Input
    X: ndarray, shape (n_channels, n_timepoints, n_trials)
    fres: int, frequency resolution
    method: str, 'WELCH' or 'MT' (multi-taper)
    window: int, window length (default: min(time series length, 2*fres))
    noverlap: int, window overlap size (default: window/2)
    nw: float, time-bandwidth product (multi-taper only)
    ntapers : int, number of tapers (multi-taper only)
        
    Output
    S: ndarray, cross-power spectral density matrix, shape (n_channels, n_channels, n_freqs)
    '''
    
    n, m, N = X.shape
    X = demean(X)
    
    # MATLAB permute([2 1 3])
    # now shape (time, channels, trials)
    X = numpy.transpose(X, (1,0,2))  
    
    nfft = 2 * fres
    if window is None:
        window = min(m, nfft)
    if noverlap is None:
        noverlap = round(window / 2)
    
    if ntapers is None:
        ntapers = int(2*nw - 1)
    
    if method.upper() == 'MT':
        S = 0
        # shape (ntapers, window)
        tapers = windows.dpss(window, nw, Kmax=ntapers, return_ratios=False)
        # shape (window, ntapers, 1)
        tapers = tapers.T[:, :, numpy.newaxis]
        nchunks = int(numpy.floor((m - noverlap) / (window - noverlap)))
        
        for r in range(N):
            S += cpsd_mt(X[:,:,r], n, fres+1, window, noverlap, nchunks, tapers)
        
        S = numpy.transpose(S, (1,2,0)) / N
        
    elif method.upper() == 'WELCH':
        S = 0
        for r in range(N):
            S += cpsd_welch(X[:,:,r], n, fres+1, window, noverlap)
        S = numpy.pi * S / N
    else:
        raise ValueError(f"Unknown method '{method}'")
    
    # fill lower triangular with conjugate
    for i in range(n):
        for j in range(i+1, n):
            S[j,i,:] = numpy.conj(S[i,j,:])
    
    return S


def cpsd_mt(X, n, h, window, noverlap, nchunks, taparray):
    nfft = 2*(h-1)
    S = numpy.zeros((h, n, n), dtype=complex)
    winstep = window - noverlap
    #ntapers = taparray.shape[1]
    
    for k in range(nchunks):
        idx = slice(k*winstep, k*winstep + window)
        XSEG = X[idx, :]
        
        # apply tapers: (window, ntapers, n_channels)
        P = numpy.fft.fft(taparray * XSEG[:, numpy.newaxis, :], n=nfft, axis=0)
        P = P[:h, :, :]
        
        for i in range(n):
            for j in range(i, n):
                S[:,i,j] += numpy.mean(P[:,:,i] * numpy.conj(P[:,:,j]), axis=1)
    
    return S / nchunks


def cpsd_welch(X, n, h, window, noverlap):
    nfft = 2*(h-1)
    S = numpy.zeros((n,n,h), dtype=complex)
    
    for i in range(n):
        f, Pxx = welch(X[:,i], nperseg=window, noverlap=noverlap, nfft=nfft)
        S[i,i,:] = Pxx
        for j in range(i+1, n):
            f, Pxy = csd(X[:,i], X[:,j], nperseg=window, noverlap=noverlap, nfft=nfft)
            S[i,j,:] = Pxy
    
    return S




def cpsd_welch_matlab(X, n, h, nfft, window, noverlap, fs=1.0):
    '''
    MATLAB-style cross-power spectral density using Welch method.

    # matlab cpsd computes cpsd using a window, overlap, and FFT length.
    # scipy.signal.csd in Python does same, with similar arguments (nperseg, noverlap, nfft, window)

    ### Scaling
    #MATLAB cpsd returns values normalized by sampling frequency and the window so that integrating the PSD gives signal variance.
    #SciPy csd normalizes, but scaling can differ depending on the scaling argument ('density' or 'spectrum').

    #density = power per Hz (like MATLAB’s default).
    #spectrum = power, but not per Hz (slightly different).

    ### Windowing 
    # MATLAB: default hamming window.
    #SciPy: default hann window. Set window='hamming'

    # same:
    # window type and length
    # nfft / frequency resolution
    # scaling (usually 'density').
    # sampling frequency matches (fs in Python).
    
    Input
    X: ndarray, shape (time, n_channels), single-trial time series data
    n: int, number of channels
    h: int, number of frequency bins (MATLAB uses nfft=2*(h-1), I think...) 
    window : int, window length
    noverlap : int, overlap between windows
    fs: float, sampling frequency (default 1), necessary for exact frequency axis

    Output
    S: ndarray, shape (n, n, h), cross-power spectral density matrix
    '''

    # definition used by MATLAB
    #nfft = 2*(h-1)
    #window = min([X.shape[0], nfft])
    #window = int(X.shape[0] / 2)
    #noverlap = window // 2    # 50% overlap
    #noverlap = 30

    S = numpy.zeros((n, n, h), dtype=complex)
    
    # MATLAB uses 'hamming' window by default
    # or (pwelch)?
    win = windows.hamming(window, sym=False)
    
    for i in range(n):
        # scaling='density' ==> matches MATLAB’s PSD units
        # return_onesided=True ==> MATLAB uses one-sided PSD by default
        f, Pxx = welch(X[:,i], fs=fs, window=win, nperseg=window, noverlap=noverlap, nfft=nfft, return_onesided=True, scaling='density')
        S[i,i,:] = Pxx
        for j in range(i+1, n):
            f, Pxy = csd(X[:,i], X[:,j], fs=fs, window=win, nperseg=window, noverlap=noverlap, nfft=nfft, return_onesided=True, scaling='density')
            S[i,j,:] = Pxy
    
    # lower triangle with complex conjugates ==>  matches MATLAB CPSD output
    # [i, j, :] slice of S is CPSD between i and j
    for i in range(n):
        for j in range(i+1, n):
            S[j,i,:] = numpy.conj(S[i,j,:])
    
    return S




def compute_integrated_coherence(S, freqs):
    '''
    Compute the coherence matrix and its integral.
    cpsd_eco.m
    
    Input
    S: numpy.ndarray, cross-power spectral density (channels, channels, n_freqs)
    freqs: numpy.ndarray, frequency vector
    
    Output
    int_cohMat_welch: numpy.ndarray, integrated coherence (single-sided)
    int_cohMat_welch_two_side : numpy.ndarray, integrated coherence (two-sided)
    '''
    n_channels = S.shape[0]
    cohMat_welch = numpy.zeros_like(S, dtype=float)
    
    for i in range(n_channels):
        for j in range(n_channels):
            cohMat_welch[i,j,:] = numpy.abs(S[i,j,:])**2 / (numpy.abs(S[i,i,:]) * numpy.abs(S[j,j,:]))
    
    df = numpy.mean(numpy.diff(freqs))
    int_cohMat_welch = numpy.sum(cohMat_welch, axis=2) * df
    int_cohMat_welch_two_side = (numpy.sum(cohMat_welch[:,:,1:], axis=2) * 2 + cohMat_welch[:,:,0]) * df
    
    return int_cohMat_welch, int_cohMat_welch_two_side








def phase_randomized_coherence_null(x, y, fs=1.0, window=None, noverlap=None, nfft=None, n_surr=1000):

    #n, h, fs=1.0
    """
    Generate a null distribution for magnitude-squared coherence between two signals
    using phase-randomized surrogates of x.

    Parameters
    ----------
    x, y : ndarray, shape (n_samples,)
        Input time series (single trial)
    fs : float
        Sampling frequency (default 1.0)
    window : int or None
        Window length for Welch (default: entire signal)
    noverlap : int or None
        Overlap for Welch
    nfft : int or None
        FFT length for Welch
    n_surr : int
        Number of surrogate iterations

    Returns
    -------
    freqs : ndarray
        Frequency vector
    C_null : ndarray, shape (n_surr, n_freqs)
        Surrogate coherence values
    C_thresh : ndarray
        95th percentile threshold at each frequency
    """

    n_samples = len(x)
    if window is None:
        window = n_samples
    if noverlap is None:
        noverlap = window // 2
    if nfft is None:
        nfft = max(256, window)  # at least 256 or window length

    # Compute real coherence once for frequency vector
    freqs, _ = welch(x, fs=fs, nperseg=window, noverlap=noverlap, nfft=nfft)
    n_freqs = len(freqs)
    C_null = numpy.zeros((n_surr, n_freqs))

    for k in range(n_surr):
        # Phase-randomize x
        Xf = numpy.fft.fft(x)
        N = len(x)
        phases = numpy.angle(Xf)

        # DC and Nyquist remain unchanged
        idx = numpy.arange(1, N//2)
        random_phases = numpy.random.uniform(-numpy.pi, numpy.pi, len(idx))
        phases[idx] = random_phases
        phases[-idx] = -random_phases[::-1]  # preserve conjugate symmetry

        Xf_surr = numpy.abs(Xf) * numpy.exp(1j*phases)
        x_surr = numpy.fft.ifft(Xf_surr).real

        # Compute magnitude-squared coherence
        f, Pxx = welch(x_surr, fs=fs, nperseg=window, noverlap=noverlap, nfft=nfft)
        f, Pxy = csd(x_surr, y, fs=fs, nperseg=window, noverlap=noverlap, nfft=nfft)
        f, Pyy = welch(y, fs=fs, nperseg=window, noverlap=noverlap, nfft=nfft)

        C_null[k, :] = numpy.abs(Pxy)**2 / (Pxx * Pyy)

    # 95th percentile threshold at each frequency
    C_thresh = numpy.percentile(C_null, 95, axis=0)

    return freqs, C_null, C_thresh





def coherence_null_from_cpsd(S, n_surr=1000, seed=None):
    """
    Generate a null distribution for magnitude-squared coherence using phase-randomized surrogates
    from an existing CPSD matrix S (n_channels, n_channels, n_freqs).

    Parameters
    ----------
    S : ndarray, shape (n_channels, n_channels, n_freqs)
        Cross-power spectral density matrix
    n_surr : int
        Number of surrogate iterations
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    C_thresh : ndarray, shape (n_channels, n_channels, n_freqs)
        95th percentile coherence threshold at each frequency
    """

    rng = numpy.random.default_rng(seed)
    n_channels, _, n_freqs = S.shape
    C_null = numpy.zeros((n_surr, n_channels, n_channels, n_freqs))

    # Compute magnitude spectrum for each channel
    #Amp = numpy.sqrt(numpy.abs(numpy.diagonal(S, axis1=0, axis2=1)))  # shape: (n_channels, n_freqs)
    Amp = numpy.sqrt(numpy.abs(numpy.diagonal(S, axis1=0, axis2=1)).T)

    for k in range(n_surr):
        # Generate phase-randomized surrogates for each channel
        phases = rng.uniform(-numpy.pi, numpy.pi, size=(n_channels, n_freqs))
        # DC component (freq=0) should have 0 phase
        phases[:, 0] = 0
        # Nyquist if even number of freqs (last index) should also have 0 phase
        if n_freqs % 2 == 0:
            phases[:, -1] = 0

        # Construct surrogate CPSD matrix assuming **no coherence** (random phase)
        for i in range(n_channels):
            for j in range(i, n_channels):
                # S_ij = Amp_i * Amp_j * exp(i*(phi_i - phi_j))
                S_surr = Amp[i,:] * Amp[j,:] * numpy.exp(1j * (phases[i,:] - phases[j,:]))
                C_null[k, i, j, :] = numpy.abs(S_surr)**2 / (Amp[i,:]**2 * Amp[j,:]**2)
                if i != j:
                    C_null[k, j, i, :] = C_null[k, i, j, :]  # symmetric

    # 95th percentile across surrogates
    C_thresh = numpy.percentile(C_null, 95, axis=0)

    return C_thresh



def phase_randomized_coherence_null(x, y, fs=1.0, window=64, noverlap=None, nfft=None, n_surr=1000, seed=None):
    '''
    Compute null distribution of magnitude-squared coherence via phase randomization.

    Returns
    freqs: frequency vector
    C_null: array of shape (n_surr, n_freqs)
    C_thresh: 95% threshold at each frequency
    '''

    if seed is not None:
        numpy.random.seed(seed)
    
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    n_samples = len(x)
    noverlap = window // 2 if noverlap is None else noverlap
    nfft = nfft or max(256, window)
    
    # Compute frequency vector
    freqs, _ = welch(x, fs=fs, window='hamming', nperseg=window, noverlap=noverlap, nfft=nfft)
    n_freqs = len(freqs)
    
    C_null = numpy.zeros((n_surr, n_freqs))
    
    X_fft = numpy.fft.fft(x)
    for k in range(n_surr):
        # Randomize phases (except DC and Nyquist)
        phases = numpy.angle(X_fft)
        amp = numpy.abs(X_fft)
        random_phases = numpy.random.uniform(0, 2*numpy.pi, n_samples//2 - 1)
        phases[1:n_samples//2] = random_phases
        phases[-(n_samples//2)+1:] = -random_phases[::-1]  # Hermitian symmetry
        X_surr = numpy.fft.ifft(amp * numpy.exp(1j * phases)).real
        
        # Compute coherence with y
        f, Cxy = csd(X_surr, y, fs=fs, window='hamming', nperseg=window, noverlap=noverlap, nfft=nfft, scaling='density', return_onesided=True)
        f, Pxx = welch(X_surr, fs=fs, window='hamming', nperseg=window, noverlap=noverlap, nfft=nfft, scaling='density', return_onesided=True)
        f, Pyy = welch(y, fs=fs, window='hamming', nperseg=window, noverlap=noverlap, nfft=nfft, scaling='density', return_onesided=True)
        
        C_null[k,:] = numpy.abs(Cxy)**2 / (Pxx * Pyy)
    
    C_thresh = numpy.percentile(C_null, 95, axis=0)
    
    return freqs, C_null, C_thresh



def lag_null_distribution(x, y, S_shape, freqs, nfft, window, noverlap, fs, n_surr=1000, min_coh_xy=0.3, seed=123456789):

    numpy.random.seed(seed)
    
    n_samples = len(x)
    time_lags = []

    X_fft = numpy.fft.fft(x)
    
    for k in range(n_surr):
        # Randomize phases
        phases = numpy.angle(X_fft)
        amp = numpy.abs(X_fft)
        random_phases = numpy.random.uniform(0, 2*numpy.pi, n_samples//2 - 1)
        phases[1:n_samples//2] = random_phases
        phases[-(n_samples//2)+1:] = -random_phases[::-1]  # Hermitian symmetry
        x_surr = numpy.fft.ifft(amp * numpy.exp(1j * phases)).real

        # Compute CPSD (can reuse your cpsd_welch_matlab function)
        S_surr = cpsd_welch_matlab(numpy.column_stack([x_surr, y]), n=2, h=S_shape[2], nfft=nfft, window=window, noverlap=noverlap, fs=fs)
        S_xy_surr = S_surr[0,1,:]
        
        # Phase & time lag
        phase_xy_surr = numpy.angle(S_xy_surr)
        freqs_nonzero = freqs.copy()
        freqs_nonzero[0] = numpy.nan
        time_lag_surr = phase_xy_surr / (2*numpy.pi*freqs_nonzero)

        # Weighted average over strong coherence
        coh_surr = numpy.abs(S_xy_surr)**2 / (S_surr[0,0,:] * S_surr[1,1,:])
        mask = coh_surr > 0.3
        avg_lag_surr = numpy.nanmean(time_lag_surr[mask])
        #avg_lag_surr = numpy.nanmean(time_lag_surr)
        time_lags.append(avg_lag_surr)

    return numpy.array(time_lags)



if __name__ == "__main__":

    param_dict =  pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, 'rb'))

    otu = 'Otu000001'
    otu_idx = param_dict['otu_labels'].index(otu)

    #days_dna = param_dict['data']['days']['DNA'][otu_idx]
    afd_dna = numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_idx])

    #ays_rna = param_dict['data']['days']['RNA'][otu_idx]
    afd_rna = numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_idx])

    #rna_dna = numpy.concatenate([afd_rna, afd_dna])

    #rna_dna = numpy.stack((afd_rna, afd_dna), axis=1)[:, :, numpy.newaxis]
    rna_dna = numpy.column_stack((afd_rna, afd_dna))


    #S, freqs = cpsd_welch_matlab(rna_dna, fres=fres, fs=fs)

    n = rna_dna.shape[1]        # number of channels
    fres = 128                # frequency resolution
    h = fres + 1              # MATLAB’s h = fres + 1
    #window = 256              # window length
    fs=1.0
    nfft = 2*(h-1)
    window = int(rna_dna.shape[0] / 2)
    noverlap = 30
    

    S = cpsd_welch_matlab(rna_dna, n=n, h=h, nfft=nfft, window=window, noverlap=noverlap, fs=1.0)

    # lag

    S_xy = S[0,1,:]

    # Phase spectrum (radians)
    phase_xy = numpy.angle(S_xy)

    # Avoid division by zero at DC
    freqs = numpy.linspace(0, fs/2, h)
    freqs_nonzero = freqs.copy()
    freqs_nonzero[0] = numpy.nan  # ignore 0 Hz for lag calculation

    # Time lag at each frequency (same units as 1/fs, e.g., weeks)
    time_lag = phase_xy / (2 * numpy.pi * freqs_nonzero)

    min_coh_xy = 0.3

    # magnitude-squared coherence
    coh_xy = numpy.abs(S_xy)**2 / (S[0,0,:] * S[1,1,:])
    mask = coh_xy > min_coh_xy
    avg_lag = numpy.nanmean(time_lag[mask])
    #avg_lag = numpy.nanmean(time_lag)
    
    print(avg_lag)

    time_lags_null = lag_null_distribution(afd_rna, afd_dna, S.shape, freqs, nfft=nfft, window=window, n_surr=500, min_coh_xy=min_coh_xy, seed=123456789)

    print(time_lags_null)

    #freqs = numpy.linspace(0, fs/2, h)
    #int_cohMat_welch, int_cohMat_welch_two_side = compute_integrated_coherence(S, freqs)
    # 2x2 matrix
    #int_cohMat_welch_rna_dna = int_cohMat_welch[0,1]

    #print(int_cohMat_welch)

    #freqs, C_null, C_thresh = phase_randomized_coherence_null(afd_rna, afd_dna, fs=fs, window=window, noverlap=noverlap, nfft=nfft, n_surr=1000)
    #C_thresh = coherence_null_from_cpsd(S, n_surr=1000, seed=123456789)
    #freqs, C_null, C_thresh = phase_randomized_coherence_null(afd_rna, afd_dna, fs=fs, window=window, noverlap=noverlap, nfft=nfft, n_surr=1000, seed=123456789)

    # integrate coherence across frequencies to match int_coh_obs
    #df = numpy.mean(numpy.diff(freqs))
    #int_C_null = numpy.sum(C_null, axis=1) * df

    #alpha=0.05
    #int_C_thresh = numpy.percentile(int_C_null, 100*(1-alpha))

    #print(int_cohMat_welch_rna_dna, int_C_thresh)

    # null
    #Take the FFT of one signal.

    #Randomize the phases of all Fourier components (except DC and Nyquist, if present).

    #Inverse FFT → surrogate time series with same amplitude spectrum but random phase.

    #Compute coherence with the other signal.

    #Repeat → null distribution.


    #Preserves the power spectrum of the signal (so your surrogate has the same variance at all frequencies).

    #Destroys the phase relationship → removes true coherence.

    #Produces frequency-dependent thresholds.

    #This is exactly what neuroscientists use for short, oscillatory signals.


