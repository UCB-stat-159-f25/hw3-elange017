import numpy as np
# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


# make wav (sound) files from the whitened data, +-2s around the event.

from scipy.io import wavfile

# function to keep the data within integer limits, and write to wavfile:
def write_wavfile(filename,fs,data):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename,int(fs), d)

# function that shifts frequency of a band-passed signal
def reqshift(data,fshift=100,sample_rate=4096):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    y[0:nbins]=0.
    z = np.fft.irfft(y)
    return z

import matplotlib.pyplot as plt

def plot_matched_filter_results(time, timemax, tevent, eventname, det, SNR, 
                                strain_whitenbp, template_match, template_fft,
                                datafreq, data_psd, plottype='png'):
    """
    Plot matched filter results for a detector and save figures.
    """

    # -- Plot SNR vs time
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-timemax, SNR, 'r' if det=='H1' else 'g', label=f'{det} SNR(t)')
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.legend(loc='upper left')
    plt.title(f'{det} matched filter SNR around event')

    plt.subplot(2,1,2)
    plt.plot(time-timemax, SNR, 'r' if det=='H1' else 'g', label=f'{det} SNR(t)')
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlim([-0.15,0.05])
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.legend(loc='upper left')
    plt.savefig(f'figures/{eventname}_{det}_SNR.{plottype}')

    # -- Plot whitened strain and template
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-tevent, strain_whitenbp, 'r' if det=='H1' else 'g', label=f'{det} whitened h(t)')
    plt.plot(time-tevent, template_match, 'k', label='Template(t)')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.ylabel('Whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(f'{det} whitened data around event')

    plt.subplot(2,1,2)
    plt.plot(time-tevent, strain_whitenbp-template_match, 'r' if det=='H1' else 'g', label=f'{det} resid')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.ylabel('Whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(f'{det} Residual whitened data after subtracting template around event')
    plt.savefig(f'figures/{eventname}_{det}_matchtime.{plottype}')

    # -- PSD and template plot
    plt.figure(figsize=(10,6))
    template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq)) 
    plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(datafreq, np.sqrt(data_psd), 'r' if det=='H1' else 'g', label=f'{det} ASD')
    plt.xlim(20, datafreq.max())
    plt.ylim(1e-24, 1e-20)
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(f'{det} ASD and template around event')
    plt.savefig(f'figures/{eventname}_{det}_matchfreq.{plottype}')
