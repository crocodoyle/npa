import fooof, npa

from fooof.sim.gen import gen_power_spectrum

freq_range = [3, 40]
aperiodic_params = [1, 1]
gaussian_params = [10, 0.3, 1]

fs, ps = gen_power_spectrum(freq_range, aperiodic_params, gaussian_params)

def test_function_defaults():
    ff = fooof.FOOOF()
    ff.fit(fs, ps, freq_range)

    amp = npa.NPA(ff, 100)

    amp.fit_log_filters()
    amp.fit_peak_filters()
