import numpy as np
from bisect import bisect_left
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel
import pandas as pd
import pickle
import os
import scipy

from sklearn.linear_model import LogisticRegression


class LineProfiles:
    
    '''
    Class to fit Voigt profiles to the Balmer absorption lines stellar spectra.
    Line profiles are fit using the LMFIT package via chi^2 minimization. 
    '''

    def __init__(self, verbose = False, plot_profiles = False, 
                 lines = ['alpha', 'beta', 'gamma', 'delta'], optimizer = 'leastsq'):

        self.verbose = verbose
        self.optimizer = optimizer
        self.halpha = 6564.61
        self.hbeta = 4862.68
        self.hgamma = 4341.68
        self.hdelta = 4102.89
        self.plot_profiles = plot_profiles
        self.lines = lines
        self.linedict = dict(alpha = self.halpha, beta = self.hbeta, gamma = self.hgamma, delta = self.hdelta)
        self.window_dict = dict(alpha = 400, beta = 400, gamma = 150, delta = 75)
        self.edge_dict = dict(alpha = 200, beta = 200, gamma = 75, delta = 65)

        self.features = [];
        self.line_ident = '_'
        self.fit_params = ['amp', 'center', 'sigma', 'gamma', 'fwhm', 'height']
        for linename in lines:
            self.features.append(linename[0] + '_fwhm')
            self.features.append(linename[0] + '_height')
            self.line_ident = self.line_ident + linename[0]


    def linear(self, wl, p1, p2):
        return p1 + p2*wl

    def chisquare(self, residual):
        return np.sum(residual**2)

    def fit_line(self, wl, flux, centroid, window = 400, edges = 200, make_plot = False):
        '''
        Fit a Voigt profile around a specified centroid on the spectrum. 

        The continuum is normalized at each absorption line via a simple linear polynimial through the edges.
        Window size and edge size can be modified. 
        
        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        flux : array
            Flux array of spectrum
        centroid : float
            The theoretical centroid of the absorption line that is being fitted, in wavelength units. 
        window : float, optional
            How many Angstroms away from the line centroid are included in the fit 
            (in both directions). This should be large enough to include the absorption line as well as 
            some continuum on either side.
        edges : float, optional
            What distance in Angstroms around each line (measured from the line center outwards)
            to exclude from the continuum-fitting step. This should be large enough to cover most of the 
            absorption line whilst leaving some continuum intact on either side. 
        make_plot : bool, optional
            Make a plot of the fit. 
        Returns
        -------
            lmfit `result` object
            A `result` instance from the `lmfit` package, from which fitted parameters and fit statistics can be extracted. 

        '''

        in1 = bisect_left(wl,centroid-window)
        in2 = bisect_left(wl,centroid+window)
        cropped_wl = wl[in1:in2]
        cropped_flux = flux[in1:in2]

        cmask = (cropped_wl < centroid - edges)+(cropped_wl > centroid + edges)

        p,cov = curve_fit(self.linear,cropped_wl[cmask],cropped_flux[cmask])

        continuum_normalized = 1 - (cropped_flux / self.linear(cropped_wl, p[0], p[1]))
        
        voigtfitter = VoigtModel()
        params = voigtfitter.make_params()
        params['amplitude'].set(min = 0,max = 100,value = 25)
        params['center'].set(value = centroid, max = centroid + 25, min = centroid - 25)
        params['sigma'].set(min = 0, max=200, value=10, vary = True)
        params['gamma'].set(value=10, min = 0, max=200, vary = True)

        try:

            result = voigtfitter.fit(continuum_normalized, params, x = cropped_wl, nan_policy = 'omit', method=self.optimizer)
        except:
            print('line profile fit failed! make sure the selected line is present on the provided spectrum')
            raise

        if make_plot:
            plt.figure(figsize = (6,3), )
            plt.plot(cropped_wl,1-continuum_normalized, 'k')
            plt.plot(cropped_wl,1-voigtfitter.eval(result.params, x = cropped_wl),'r')
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.ylabel('Normalized Flux')
            if centroid == self.halpha:
                plt.title(r'H$\alpha$')
            elif centroid == self.hbeta:
                plt.title(r'H$\beta$')
            elif centroid == self.hgamma:
                plt.title(r'H$\gamma$')
            elif centroid == self.hdelta:
                plt.title(r'H$\delta$')
            plt.show()

        return result

    def fit_balmer(self, wl, flux, make_plot = False):

        '''
        Fits Voigt profiles to the first three Balmer lines (H-alpha, H-beta, and H-gamma). Returns all 18 fitted parameters. 
        
        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        flux : array
            Flux array of spectrum
        make_plot : bool, optional
            Plot all individual Balmer fits. 

        Returns
        -------
            array
            Array of 18 Balmer parameters, 6 for each line. If the profile fit fails, returns array of 18 `np.nan` values. 

        '''
        colnames = [];
        parameters = [];
        for linename in self.lines:
            colnames.extend([linename[0] + '_' + fparam for fparam in self.fit_params])
            try:
                line_parameters = np.asarray(self.fit_line(wl, flux, self.linedict[linename], 
                                                           self.window_dict[linename], self.edge_dict[linename], 
                                                           make_plot = make_plot).params)
                parameters.extend(line_parameters)
            except KeyboardInterrupt:
                raise
            except:
                raise
                print('profile fit failed! returning NaN...')
                parameters.extend(np.repeat(np.nan, 6))


        balmer_parameters = pd.DataFrame([parameters], columns = colnames)

        return balmer_parameters
    
class LRClassifier:
    def __init__(self, features = ['Su', 'Sg', 'Sr', 'Si', 'Sz' , 'a_fwhm', 'a_height',
                                               'b_fwhm', 'b_height',
                                               'g_fwhm', 'g_height',
                                               'd_fwhm', 'd_height'], training_grid = 'training_grid.csv'
                  ):
        
        self.features = features
        
        self.training_grid = pd.read_csv(training_grid)
        
        seds = [];
        balmer_features = [];
        for feature in features:
            if '_' not in feature:
                seds.append(self.training_grid[feature])
            elif '_' in feature:
                balmer_features.append(self.training_grid[feature])
        
        seds = np.array(seds).T
        self.balmer_features = np.array(balmer_features).T
        
        colors = [];
        self.ncols = seds.shape[1]
        for i in range(self.ncols):
            for j in np.arange(i+1, self.ncols):
                colors.append(seds[:, i] - seds[:, j])

        self.colors = np.array(colors).T
        
        self.lr = LogisticRegression(penalty = 'l2', solver = 'saga', max_iter = 1e5)
        
        if len(self.balmer_features) > 0:
            self.all_features = np.hstack((self.colors, self.balmer_features))
        else: 
            self.all_features = self.colors
        
        selected_grid = np.array(self.all_features)
        target = np.array(self.training_grid['is_mpms'])
        
        self.lr.fit(selected_grid, target)
                
    def classify(self, X):
        
        seds = X[:, :self.ncols]
        colors = [];
        for i in range(self.ncols):
            for j in np.arange(i+1, self.ncols):
                colors.append(seds[:, i] - seds[:, j])
        colors = np.array(colors).T
        
        data = np.hstack((colors, X[:, self.ncols:]))
        
        
        return self.lr.predict_proba(data)[:, 1]