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
def reqshift(data, fshift =100, sample_rate = 4096):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    y[0:nbins] = 0.
    z = np.fft.irfft(y)
    return z

import matplotlib.pyplot as plt
import os

def plot_matched_filter_results(time, timemax, tevent, eventname, det, SNR, strain_whitenbp,
                                template_match, template_fft,
                                datafreq, data_psd, plottype='png', make_dir='figures'):
    os.makedirs(make_dir, exist_ok=True)

    pcolor = 'g' if det == 'L1' else 'r'
    
    # --- Plot SNR time series
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time - timemax, SNR, pcolor, label=f'{det} SNR(t)')
    plt.grid(True)
    plt.ylabel('SNR')
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.legend(loc='upper left')
    plt.title(f'{det} matched filter SNR around event')

    plt.subplot(2,1,2)
    plt.plot(time - timemax, SNR, pcolor, label=f'{det} SNR(t)')
    plt.grid(True)
    plt.ylabel('SNR')
    plt.xlim([-0.15, 0.05])
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.legend(loc='upper left')
    plt.savefig(f'{make_dir}/{eventname}_{det}_SNR.{plottype}')
    plt.close()

    # --- Plot whitened strain and template
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time - tevent, strain_whitenbp, pcolor, label=f'{det} whitened h(t)')
    plt.plot(time - tevent, template_match, 'k', label='Template(t)')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid(True)
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.ylabel('whitened strain (noise σ units)')
    plt.legend(loc='upper left')
    plt.title(f'{det} whitened data around event')

    plt.subplot(2,1,2)
    plt.plot(time - tevent, strain_whitenbp - template_match, pcolor, label=f'{det} resid')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid(True)
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.ylabel('whitened strain (noise σ units)')
    plt.legend(loc='upper left')
    plt.title(f'{det} Residual whitened data after subtracting template')
    plt.savefig(f'{make_dir}/{eventname}_{det}_matchtime.{plottype}')
    plt.close()

    # --- Plot PSD and template
    # Use only the positive frequency part of the FFT
    pos_freqs = datafreq[:len(data_psd)]
    template_f = np.abs(template_fft[:len(pos_freqs)]) * np.sqrt(np.abs(pos_freqs))
    plt.loglog(pos_freqs, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(pos_freqs, np.sqrt(data_psd), pcolor, label=f'{det} ASD')
    plt.xlim(20, 2048)
    plt.ylim(1e-24, 1e-20)
    plt.grid(True)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz)')
    plt.legend(loc='upper left')
    plt.title(f'{det} ASD and template around event')
    plt.savefig(f'{make_dir}/{eventname}_{det}_matchfreq.{plottype}')
    plt.close()
