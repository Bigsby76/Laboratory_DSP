#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.signal
import scipy.fftpack

import numpy as np
from matplotlib import pyplot as plt


fs_rate, signal = wavfile.read("LAB5_500HzFHR.wav")
print ("Frequency sampling", fs_rate)
l_audio = len(signal.shape)
print ("Channels", l_audio)
if l_audio == 2:
    signal = signal.sum(axis=1) / 2
N = signal.shape[0]
print ("Complete Samplings N", N)
secs = N / float(fs_rate)
print ("secs", secs)
Ts = 1.0/fs_rate # sampling interval in time
print ("Timestep between samples Ts", Ts)
t = np.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
FFT_full = abs(scipy.fft.fft(signal))
FFT_side = FFT_full[range(N//2)] # one side FFT range
freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(N//2)] # one side frequency range
fft_freqs_side = np.array(freqs_side)

#Filtering Low pqss filter
b, a = scipy.signal.butter(10, 0.02, 'low')
output_signal = scipy.signal.filtfilt(b, a, signal)
#fft of the filtered signal
FFT_full_filter = abs(scipy.fft.fft(output_signal))
FFT_side_filter = FFT_full_filter[range(N//2)] # one side FFT range



plt.subplot(511)
p1 = plt.plot(t, signal, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(512)
p2 = plt.plot(freqs, FFT_full, "r") # plotting the complete fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
plt.subplot(513)
p3 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')
plt.subplot(514)
p4 = plt.plot(t, output_signal, "g") # plotting the filtered signal
plt.xlabel('Time')
plt.ylabel('Amplitude filtered')
plt.subplot(515)
p5 = plt.plot(freqs_side, FFT_side_filter, "b") # plotting the filtered signal
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided filtered signal')
plt.show()