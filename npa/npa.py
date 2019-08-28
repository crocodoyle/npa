# Author: Andrew Doyle <andrew.doyle@mcgill.ca>
#
# License: Apache 2.0

import numpy as np
from scipy.signal import cheby1, remez, filtfilt, butter, freqz, firwin2

from mne.parallel import parallel_func
from functools import partial


class NPA(object):
    """Learn and apply filters on components of neural power spectra

    Parameters
    ----------
    fooof : object of type FOOOF
        Parametric model of the power spectrum
    sampling_frequency : int
        Sampling frequency of time series to be amplified
    log_approx_levels : int, optional
        Number of filters to use to approximate logarithmic 1/f filter response
    n_peak_taps : int, optional
        Number of taps to use in the FIR filter to select spectral peaks

    Attributes
    ----------
    log_filter_coeffs : list of [1d array, 1d array]
        Filter coefficients [b, a] for each stage of the logarithmic filter approximation
    log_filter_amplitudes = list of float
        Amplitudes to multiply outputs of each stage of the 1/f negating filter

    peak_filter_coeffs = list of [1d array, 1d array]
        Filter coefficients [b, a] to select each peak of the neural power spectrum
    peak_filter_amplitudes = list of float
        Amplitudes to multiply outputs of peak-selecting filters
    """
    nyquist = 0
    sampling_frequency = 0

    log_approx_levels = 0
    n_peak_taps = 0

    #FOOOF parameters
    offset, knee, slope = 0, 0, 0
    fooof = None

    log_filter_coeffs, log_filter_amplitudes = [], []
    peak_filter_coeffs, peak_filter_amplitudes = [], []
    peak_multipliers = []

    def __init__(self, fooof, sampling_frequency, log_approx_levels=5, n_peak_taps=4):
        self.fooof = fooof

        self.sampling_frequency = sampling_frequency
        self.nyquist = sampling_frequency / 2

        self.log_approx_levels = log_approx_levels
        self.peak_mode = None
        self.n_peak_taps = n_peak_taps

        n_freqs = 1000
        self.peak_actual_gain = np.zeros(n_freqs)
        self.peak_ideal_gain = np.zeros(n_freqs)

        self.log_actual_gain = np.zeros(n_freqs)
        self.log_ideal_gain = np.zeros(n_freqs)

        if fooof.aperiodic_mode == 'knee':
            self.slope = fooof.aperiodic_params_[2]
            self.knee = fooof.aperiodic_params_[1]
        else:
            self.slope = fooof.aperiodic_params_[1]
            self.knee = 0.
        if self.knee < 0.:
            self.knee = 0.
        self.offset = fooof.aperiodic_params_[0]


    def fit_log_filters(self, n_stages=5):
        '''Calculates the system of IIR digital filters that approximates the 1/f component

        Parameters
        ----------
        n_stages : int
            Number of filters to use to approximate logarithmic 1/f filter response
        '''
        self.log_filter_coeffs = []
        self.log_filter_amplitudes = []

        for i in range(n_stages):
            L_half = 1 - (1 / (2 ** (i + 1)))  # Half of remaining Log amplitude
            f_half = ((self.knee + self.nyquist ** self.slope) ** L_half - self.knee) ** (1 / self.slope)  # Frequency of half of remaining log amp

            max_ripple = 0.4 * np.sqrt((2 ** i) * 4 * self.slope)

            b, a = cheby1(1, max_ripple, f_half, btype='highpass', fs=self.sampling_frequency)

            coeffs = tuple((b, a))

            self.log_filter_coeffs.append(coeffs)
            self.log_filter_amplitudes.append(1 - L_half)


    def fit_peak_filters(self, mode='sharp', n_taps=256):
        '''Calculates the FIR filter that selects each peak

        Parameters
        ----------
        mode : str
            Whether to use a boxcar bandpass filter ('normal') or a filter with an approximately Gaussian response ('sharp')
        n_taps : int
            Length of FIR filter to use for peak filter
        '''
        self.peak_mode = mode

        for idx, ([centre_frequency, amplitude, bw]) in enumerate(self.fooof.peak_params_):
            std_dev = bw / 2
            # wp = 1 / (np.sqrt(np.abs(np.log(0.25 * np.pi * std_dev ** 2) * (std_dev ** 2))))

            ws_low = (centre_frequency - 3 * std_dev)
            ws_high = (centre_frequency + 3 * std_dev)

            if ws_low < 0:
                ws_low = centre_frequency / 2

            if ws_high > self.nyquist:
                ws_high = self.nyquist - 1

            if 'normal' in mode:
                b, a = butter(n_taps, [ws_low, ws_high], btype='bandpass', fs=self.sampling_frequency)

            if 'sharp' in mode:
                ws_low = centre_frequency - 3 * std_dev
                ws_high = centre_frequency + 3 * std_dev

                wp_low = centre_frequency - std_dev
                wp_high = centre_frequency + std_dev

                print('Passband values:', ws_low, wp_low, wp_high, ws_high)

                if wp_low < 0:
                    b, a = cheby1(n_taps, 6, wp_high, 'low', fs=self.sampling_frequency)
                    a = [1.0]
                elif ws_low < 0:
                    b = remez(n_taps, [wp_low, wp_high, ws_high, self.nyquist], [1, 0], fs=self.sampling_frequency, maxiter=100)
                    a = [1.0]
                elif wp_high > self.nyquist:
                    b = remez(n_taps, [0, ws_low, wp_low, self.nyquist], [0, 1], fs=self.sampling_frequency, maxiter=100)
                    a = [1.0]
                else:
                    b = remez(n_taps, [0, ws_low, wp_low, wp_high, ws_high, self.nyquist], [0, 1, 0], fs=self.sampling_frequency, maxiter=100)
                    a = [1.0]

            coeffs = tuple((b, a))

            self.peak_filter_coeffs.append(coeffs)
            self.peak_filter_amplitudes.append(amplitude)

            # plot_filter(b, self.sampling_frequency, fscale='linear')


    def fit_filters(self, log_approx_levels=5, peak_mode='sharp', n_peak_taps=128):
        '''Fits all filters required to transform the power spectrum'''
        self.log_approx_levels = log_approx_levels
        self.peak_mode = peak_mode

        self.fit_log_filters(log_approx_levels)
        self.fit_peak_filters(mode=peak_mode, n_taps=n_peak_taps)


    def amplify(self, time_series, peak_multipliers=[1]):
        '''Applies the neural power amplifier to a time series

        Parameters
        ----------
        time_series : 2d numpy array
            Time series to be amplified

        Returns
        -------
        amplified_time_series : 2d numpy array
            Amplified time series
        '''
        if len(peak_multipliers) == 1:
            peak_multipliers = peak_multipliers * len(self.fooof.peak_params_)

        if len(peak_multipliers) != len(self.fooof.peak_params_):
            raise Exception('peak_multipliers was length {} but should be of length 1 or the same length as the number of peaks {}'.format(len(peak_multipliers), len(self.fooofpeak_params_)))

        self.peak_multipliers = peak_multipliers

        amplified_time_series = np.zeros_like(time_series, dtype='float64')

        filter_coeffs = self.log_filter_coeffs + self.peak_filter_coeffs

        filter_amplitudes = []
        filter_amplitudes.extend(self.log_filter_amplitudes)

        for amp, multiplier in zip(self.peak_filter_amplitudes, self.peak_multipliers):
            filter_amplitudes.append(amp*multiplier)

        for i, (coeffs, amplitude) in enumerate(zip(filter_coeffs, filter_amplitudes)):
            amplified_time_series += filtfilt(coeffs[0], coeffs[1], time_series) * amplitude

        return amplified_time_series


    def plot_log_filters(self):
        '''Plots of the frequency response for the filters that negate the 1/f'''
        import matplotlib.pyplot as plt

        log_fig = plt.figure()

        n_points = 1000
        frequencies = np.linspace(0, self.nyquist, n_points)

        logarg = self.knee + frequencies ** self.slope
        logarg[logarg < 1] = 1

        ideal_gain = np.log10(logarg)
        ideal_gain = ideal_gain / np.max(ideal_gain)
        ideal_gain[0] = 0
        ideal_log = np.clip(ideal_gain, 1e-20, 1)

        ideal_mag = np.zeros(n_points)
        approx_log = np.zeros(n_points)

        ideal_mag += ideal_log

        for idx, (coeffs, amplitude) in enumerate(zip(self.log_filter_coeffs, self.log_filter_amplitudes)):
            w, h = freqz(coeffs[0], coeffs[1], worN=n_points, fs=self.sampling_frequency)

            mag = np.maximum(np.abs(h), 1e-20) ** 2

            stage_approx = mag * amplitude
            approx_log += stage_approx

            if idx == 0:
                plt.plot(w, mag * amplitude, color='k', linewidth=2, zorder=4, label='Stages')
                plt.plot(w, ideal_log, color='r', linestyle='dashed', linewidth=2, label='Ideal')
            else:
                plt.plot(w, mag * amplitude, color='k', linewidth=2, zorder=4)

        plt.plot(w, approx_log, color='b', linewidth=2, label='Approximation')
        plt.ylabel('Gain')
        plt.xlabel('Frequency (Hz)')
        return log_fig

    def plot_peak_filters(self):
        '''Plots of the frequency response for the filters that select peaks'''
        import matplotlib.pyplot as plt

        peak_fig = plt.figure()

        n_points = 1000

        frequencies = np.linspace(0, self.nyquist, n_points)
        self.peak_ideal_gain, self.peak_actual_gain = np.zeros(n_points), np.zeros(n_points)

        for idx, (coeffs, [centre_frequency, amplitude, bw]) in enumerate(zip(self.peak_filter_coeffs, self.fooof.peak_params_)):
            std_dev = bw / 2

            if 'sharp' in self.peak_mode:
                ideal_gain = ((1/(std_dev*np.sqrt(2*np.pi)))*np.exp(-(1/2)*(((frequencies-centre_frequency)/std_dev)**2)))
                ideal_gain = ((ideal_gain - np.min(ideal_gain)) / (np.max(ideal_gain) - np.min(ideal_gain)))
                ideal_gain = (np.clip(ideal_gain, 1e-20, 1)**2) * amplitude
            else:
                ideal_gain = [1 if f > centre_frequency - 3*std_dev and f < centre_frequency + 3*std_dev else 0 for f in frequencies]
                ideal_gain = (np.array(ideal_gain)**2) * amplitude

            w, h = freqz(coeffs[0], coeffs[1], worN=n_points, fs=self.sampling_frequency)

            mag = np.maximum(np.abs(h), 1e-20)**2
            gaussian_gain = mag * amplitude

            plt.plot(w, gaussian_gain, 'b', linewidth=2, zorder=4, label='actual')
            plt.plot(w, ideal_gain, 'r', linewidth=2, zorder=3, linestyle='dashed', label='ideal')

            self.peak_ideal_gain += ideal_gain
            self.peak_actual_gain += gaussian_gain

        plt.ylabel('Gain')
        plt.xlabel('Frequency (Hz)')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16, shadow=True, fancybox=True)
        return peak_fig