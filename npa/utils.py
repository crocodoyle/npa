# Author: Andrew Doyle <andrew.doyle@mcgill.ca>
#
# License: Apache 2.0

from mne.preprocessing import ICA, create_eog_epochs


def blink_removal(eeg, eeg_channels, eog_channels, visualize=False):

    filter_eeg = eeg.copy()
    filter_eeg = filter_eeg.filter(1, 40, picks=eeg_channels, n_jobs=7, verbose=0)

    ica = ICA(n_components=0.98, method='extended-infomax')

    ica.fit(filter_eeg, picks=eeg_channels, decim=8, verbose=0)

    eog_average = create_eog_epochs(eeg, ch_name=filter_eeg.ch_names[eog_channels[1]], tmin=-.5, tmax=.5, l_freq=1, picks=eeg_channels, verbose=0).average()

    eeg_eog_channels = eeg_channels + eog_channels
    eog_epochs = create_eog_epochs(eeg, ch_name=filter_eeg.ch_names[eog_channels[1]], tmin=-.5, tmax=.5, l_freq=1, picks=eeg_eog_channels, verbose=0)

    eog_inds = []

    threshold = 3
    while len(eog_inds) < 1:
        threshold -= 0.05
        eog_inds, scores = ica.find_bads_eog(eog_epochs, ch_name=eeg.ch_names[eog_channels[1]], l_freq=1, threshold=threshold, verbose=0)
    eog_inds = [eog_inds[0]]

    if visualize:
        ica_eog_scores_fig = ica.plot_scores(scores, exclude=eog_inds, show=visualize)
        sources_fig = ica.plot_sources(eog_average, exclude=eog_inds, show=visualize)

        ica_properties_fig = ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 45., 'n_jobs': -1}, image_args={'sigma': 1.}, show=visualize)

        ica_excluded_fig = ica.plot_overlay(eog_average, exclude=eog_inds, show=visualize)

    ica.exclude.extend(eog_inds)
    eeg = ica.apply(eeg)

    return eeg