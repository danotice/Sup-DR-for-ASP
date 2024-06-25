import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
from sklearn import cross_decomposition
from scipy.spatial import distance
from scipy import stats
from scipy.linalg import eig

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, make_scorer, confusion_matrix, roc_auc_score


import pyispace
#import lol

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('mode.chained_assignment',None)


class Pilot():

    def __init__(self):
        self.V = []
        self.A = []
        self.B = []
        self.C = []

        # self.scaleF = None
        # self.scaleY = None
        
        # after dropping dependent features
        self.F_ = []
        self.A_ = []
        self.dropped = []
        self.d_max = 0


    def drop_dependent(self,X):
        Rank = np.linalg.matrix_rank(X)
        dropped = []
        featInd = np.arange(X.shape[1])
        
        i=X.shape[1]-1
        while i > 0 and X.shape[1] > Rank:
            rank = np.linalg.matrix_rank(X)
            if np.linalg.matrix_rank(np.delete(X, i, axis=1)) == rank:
                dropped.append(featInd[i])
                featInd = np.delete(featInd, i)
                X = np.delete(X, i, axis=1)
            i -= 1

        return sorted(dropped), X

    def fit(self, F, Y, d=0,featlabels=None, alglabels=None):
        # self.scaleF = StandardScaler().fit(F)
        # self.scaleY = StandardScaler().fit(Y)        
        # F = self.scaleF.transform(F)
        # Y = self.scaleY.transform(Y)
        
        if np.linalg.matrix_rank(F) < F.shape[1]:
            self.dropped, F = self.drop_dependent(F)
            
        Xbar = np.hstack((F, Y))
        m = F.shape[1]
        a = Y.shape[1]

        F = F.T
        Xbar = Xbar.T
        # add names to Xbar


        D,V = np.linalg.eig(np.dot(Xbar, Xbar.T))  
        if np.all(D.imag == 0):
            D = D.real
            V = V.real
        self.d_max = len(D)

        idx = np.argsort(np.abs(D))[::-1]
        if d == 0:
            d = len(idx)
        V = V[:, idx[:d]]          # top d eigenvectors
        self.B = V[:m, :]            # rows for features
        self.C = V[m:, :]          # rows for performance

        Xr = np.dot(F.T, np.linalg.pinv(np.dot(F, F.T)))
        A = np.dot(V.T, np.dot(Xbar, Xr))
        #Z = np.dot(A, F)
        #Xhat = np.vstack((np.dot(B, Z), np.dot(C, Z)))
        #error = float(np.sum((Xbar - Xhat) ** 2))
        # R2 = np.diagonal(np.corrcoef(Xbar, Xhat, rowvar=False)[:m+a, m+a:]) ** 2
        #Z = Z.T

        self.A_ = A.copy()
        self.dropped.sort()
        for c in self.dropped:
            A = np.insert(A, c, 0, axis=1)

        self.V = V
        self.A = A
        self.F_ = F.T
        
    def transform(self, F):
        # F = self.scaleF.transform(F)
        return np.dot(self.A, F.T).T
    
    def error0(self, F, Y):
        # F = self.scaleF.transform(F)
        # Y = self.scaleY.transform(Y)

        F_ = np.delete(F, self.dropped, axis=1)
        Xbar = np.hstack((F_, Y)).T
        
        Z = np.dot(self.A, F.T)
        Xhat = np.vstack((np.dot(self.B, Z), np.dot(self.C, Z)))

        return np.sum((Xbar - Xhat) ** 2)/Xhat.shape[1]
    
    def error(self, F, Y, d):
        # F = self.scaleF.transform(F)
        # Y = self.scaleY.transform(Y)

        F_ = np.delete(F, self.dropped, axis=1)
        Xbar = np.hstack((F_, Y)).T

        Z = np.dot(self.A[:d,:], F.T)
        Xhat = np.vstack((np.dot(self.B[:,:d], Z), np.dot(self.C[:,:d], Z)))

        return np.sum((Xbar - Xhat) ** 2)/Xhat.shape[1]

    def error_split(self, F, Y, d):
        # F = self.scaleF.transform(F)
        # Y = self.scaleY.transform(Y)

        F_ = np.delete(F, self.dropped, axis=1)
        
        Z = np.dot(self.A[:d,:], F.T)
        Fhat = np.dot(self.B[:,:d], Z)
        Yhat = np.dot(self.C[:,:d], Z)
        
        return np.sum((F_.T - Fhat) ** 2)/Fhat.shape[1], np.sum((Y.T - Yhat) ** 2)/Yhat.shape[1]

def pilot_CV(X,Y,skf,scale,return_best, return_errors):

    train_errors = []
    test_errors = []
    
    for train_idx, test_idx in skf:
        Xtrain = X[train_idx]
        Ytrain = Y[train_idx]
        Xtest = X[test_idx]
        Ytest = Y[test_idx]

        if scale:
            scaleX = StandardScaler().fit(Xtrain)
            scaleY = StandardScaler().fit(Ytrain)
            Xtrain = scaleX.transform(Xtrain)
            Ytrain = scaleY.transform(Ytrain)
            Xtest = scaleX.transform(Xtest)
            Ytest = scaleY.transform(Ytest)

        pilot = Pilot()
        pilot.fit(Xtrain, Ytrain, d=0)

        train_errors.append(
            {d:pilot.error(Xtrain, Ytrain, d) for d in range(1,pilot.d_max+1)}
        )
        test_errors.append(
            {d:pilot.error(Xtest, Ytest, d) for d in range(1,pilot.d_max+1)}
        )

    train_errors = pd.DataFrame(train_errors).mean()
    test_errors = pd.DataFrame(test_errors).mean()

    if return_best:
        best_d = test_errors.idxmin()

        if scale:
            scaleX = StandardScaler().fit(X)
            scaleY = StandardScaler().fit(Y)
            X = scaleX.transform(X)
            Y = scaleY.transform(Y)

        pilotB = Pilot()
        pilotB.fit(X, Y, d=best_d)
        if return_errors:
            return {'model':pilotB, 'best_d':best_d,'train':train_errors, 'test':test_errors}
        return {'model':pilotB, 'best_d':best_d}    
    return {'train':train_errors, 'test':test_errors}
    
def pilot_CVy(X,Y,skf,scale,return_best, return_errors):

    Ftrain_errors = []
    Ftest_errors = []
    Ytrain_errors = []
    Ytest_errors = []
    
    for train_idx, test_idx in skf:
        Xtrain = X[train_idx]
        Ytrain = Y[train_idx]
        Xtest = X[test_idx]
        Ytest = Y[test_idx]

        if scale:
            scaleX = StandardScaler().fit(Xtrain)
            scaleY = StandardScaler().fit(Ytrain)
            Xtrain = scaleX.transform(Xtrain)
            Ytrain = scaleY.transform(Ytrain)
            Xtest = scaleX.transform(Xtest)
            Ytest = scaleY.transform(Ytest)

        pilot = Pilot()
        pilot.fit(Xtrain, Ytrain, d=0)

        train = {d:pilot.error_split(Xtrain, Ytrain, d) for d in range(1,pilot.d_max+1)}
        test = {d:pilot.error_split(Xtest, Ytest, d) for d in range(1,pilot.d_max+1)}

        Ftrain_errors.append({k:v[0] for k,v in train.items()})
        Ytrain_errors.append({k:v[1] for k,v in train.items()})
        Ftest_errors.append({k:v[0] for k,v in test.items()})
        Ytest_errors.append({k:v[1] for k,v in test.items()})
        

    Ftrain_errors = pd.DataFrame(Ftrain_errors).mean()
    Ftest_errors = pd.DataFrame(Ftest_errors).mean()
    Ytrain_errors = pd.DataFrame(Ytrain_errors).mean()
    Ytest_errors = pd.DataFrame(Ytest_errors).mean()

    if return_best:
        best_d = Ytest_errors.idxmin()
        
        if scale:
            scaleX = StandardScaler().fit(X)
            scaleY = StandardScaler().fit(Y)
            X = scaleX.transform(X)
            Y = scaleY.transform(Y)

        pilotB = Pilot()
        pilotB.fit(X, Y, d=best_d)
        if return_errors:
            return {'model':pilotB, 'best_d':best_d,
                    'Ftrain':Ftrain_errors, 'Ftest':Ftest_errors,
                    'Ytrain':Ytrain_errors, 'Ytest':Ytest_errors}
        return {'model':pilotB, 'best_d':best_d}    
    return {'Ftrain':Ftrain_errors, 'Ftest':Ftest_errors,
                    'Ytrain':Ytrain_errors, 'Ytest':Ytest_errors}
    
    
# TODO - modify so PlotProj has a filler column in case Best not added
class InstanceSpace():
    
    def __init__(self):
        
        # attributes
        self.features = pd.DataFrame()
        self.performance = pd.DataFrame()
        self.featureNames = []
        self.algorithms = []

        self.n, self.m, self.a = 0, 0, 0 

        # standardized arrays
        self.X_s = []
        self.Y_s = []

        # centered arrays
        self.X_c = []
        self.Y_c = []
        

        # projection spaces
        self.PlotProj = pd.DataFrame() # projection of data points
        self.projections = {}   # projection objects
        self.proj = {}         # projection component names

        # training-test split
        split_data = {}
        Y_rel = []

    def fromMetadata(self, metadata, prefixes=['feature_','algo_']):
        self.featureNames = [x for x in metadata.columns if x.startswith(prefixes[0])]
        self.algorithms = [x for x in metadata.columns if x.startswith(prefixes[1])]

        self.features = metadata[self.featureNames]
        self.performance = metadata[self.algorithms]

        # put instance name in PlotProj
        #self.PlotProj['instance'] = metadata.index
        #self.PlotProj.set_index(metadata.index,inplace=True)

        self.n, self.m = self.features.shape
        self.a = len(self.algorithms)

        self.X_s = StandardScaler().fit_transform(self.features.values)
        self.Y_s = StandardScaler().fit_transform(self.performance.values)

        self.X_c = StandardScaler(with_std=False).fit_transform(self.features.values)
        self.Y_c = StandardScaler(with_std=False).fit_transform(self.performance.values)


    def getBest(self, best, axis=1):
        """Actual best performance based on performance metrics

        Args:
            expr: function with rule for best algorithm or a list type with the same length as the number of instances 
        """
        if callable(best):
            self.performance['Best'] = self.performance.apply(best,axis=axis)
            self.PlotProj['Best'] = self.performance['Best']#.values
            print('classification data available')

        elif len(best) == self.n:
            self.performance['Best'] = best
            self.PlotProj['Best'] = self.performance['Best']#.values
            print('classification data available')

        else:
            print('Cannot assign best algorithm. Check the length of the list or the function')
    
    def getRelativePerf(self, min):

        if min:
            self.Y_rel = self.performance[self.algorithms].apply(
                lambda row: (row - row.min())/row.min(), axis=1
            ).fillna(0).values
        else:
            self.Y_rel = self.performance[self.algorithms].apply(
                lambda row: (row - row.max())/row.max(), axis=1
            ).fillna(0).values
        
        print(f'Relative performance data available')

    def splitData(self, test_size, random_state, stratified = True):
        """Split data into training and test sets.

        Args:
            test_size (float): proportion of data to be in test set
            random_state (int): seed for random number generator
            stratified (bool, optional): whether to stratify the split. Defaults to True.
        """


        self.split_data = dict(zip(
            ['X_train', 'X_test', 'Y_train', 'Y_test', 'Yb_train', 'Yb_test'],
            
            train_test_split(self.X_s, self.Y_s, self.performance['Best'],
                             test_size=test_size, random_state=random_state, 
                             stratify= self.performance['Best'] if stratified else None)
    
        ))

        if len(self.Y_rel) > 0:
            self.split_data['Yr_train'], self.split_data['Yr_test'] = train_test_split(
                self.Y_rel, test_size=test_size, random_state=random_state, 
                stratify= self.performance['Best'] if stratified else None
            )
        print(f'Data split into training (size {len(self.split_data["X_train"])}) and test (size {len(self.split_data["X_test"])}) sets, stratified: {stratified}')

    def splitData2(self, test_size, random_state, stratified = True):
        """Split data into training and test sets - unscaled.

        Args:
            test_size (float): proportion of data to be in test set
            random_state (int): seed for random number generator
            stratified (bool, optional): whether to stratify the split. Defaults to True.
        """


        self.split_data = dict(zip(
            ['X_train', 'X_test', 'Y_train', 'Y_test', 'Yb_train', 'Yb_test'],
            
            train_test_split(self.features.values, self.performance[self.algorithms].values, 
                             self.performance['Best'],
                             test_size=test_size, random_state=random_state, 
                             stratify= self.performance['Best'] if stratified else None)
    
        ))
        self.split_data['X_train'] = self.split_data['X_train'].astype(float)
        self.split_data['X_test'] = self.split_data['X_test'].astype(float)

        if len(self.Y_rel) > 0:
            self.split_data['Yr_train'], self.split_data['Yr_test'] = train_test_split(
                self.Y_rel, test_size=test_size, random_state=random_state, 
                stratify= self.performance['Best'] if stratified else None
            )
        print(f'Unscaled data split into training (size {len(self.split_data["X_train"])}) and test (size {len(self.split_data["X_test"])}) sets, stratified: {stratified}')

    

    # TODO - remove best, add feat as type and a list of possible feats        
    def initPlotData(self, type='perf',feats=[]):
        """Puts all the data needed for plots in Proj dataframe.

        Args:
            type (str, optional): which data matrix to add (best, perf, feat). Defaults to 'best'.
        """

        if type=="perf":
            self.PlotProj[self.algorithms] = self.performance[self.algorithms]
            print('performance data available for visualisation')
        
        elif type=="feat":
            pass

    # 2D Plots
    def plots2D(self, methods, shape, hue=None, save=''):
        """Plots 2D projections of the instance space.

        Args:
            methods (list): list of methods to plot
            shape (tuple): shape of the subplot grid
            hue (str, optional): column name in PlotProj to use for coloring. Defaults to None.
        """
        #self.PlotProj.set_index('name',inplace=True)

        if methods == 'all':
            methods = list(self.projections.keys())

        plt.figure(figsize=(4*shape[1],3*shape[0]))
        
        for i, method in enumerate(methods):
            if method not in self.projections:
                print(f"{method} projection not available")
                continue
            
            plt.subplot(shape[0],shape[1],i+1)      
            ax = self.proj[method]      

            if len(ax) < 2:
                if hue == None:
                    sns.scatterplot(y=[0]*self.n,x=ax[0], data=self.PlotProj, alpha=0.5)
                    plt.title(method)
                    continue
                elif hue == 'Best':
                    # densiy plot grouped by best algorithm
                    sns.kdeplot(x=ax[0],data=self.PlotProj, hue='Best',fill=True)
                    plt.title(method)
                    continue
                else:
                    # plot ax vs hue
                    sns.scatterplot(x=ax[0],y=hue, data=self.PlotProj, alpha=0.5)
                    plt.title(method)
                    continue


            if i < len(methods)-1:
                sns.scatterplot(x=ax[0],y=ax[1],hue=hue, data=self.PlotProj, alpha=0.5, legend=False)
            
            else:
                sns.scatterplot(x=ax[0],y=ax[1],hue=hue, data=self.PlotProj, alpha=0.5)
                # legend outside plot to right
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=hue)
        
            plt.title(method)
            plt.xlabel('Z1')
            plt.ylabel('Z2')

        plt.subplots_adjust(wspace=0.25, hspace=0.4)
        plt.show()

        if save != '':
            plt.savefig(save, bbox_inches='tight')

    def plots2D_det(self, methods, shape,size,titles=[],keytitle='',suptitle='', hue=None, save=''):
        """Plots 2D projections of the instance space.

        Args:
            methods (list): list of methods to plot
            shape (tuple): shape of the subplot grid
            hue (str, optional): column name in PlotProj to use for coloring. Defaults to None.
        """
        #self.PlotProj.set_index('name',inplace=True)

        if methods == 'all':
            methods = list(self.projections.keys())

        plt.figure(figsize=(size[0],size[1]))

        if len(titles) != len(methods):
            print('suitable titles not provided')
            titles = methods
        
        
        for i, method in enumerate(methods):
            if method not in self.projections:
                print(f"{method} projection not available")
                continue
            
            plt.subplot(shape[0],shape[1],i+1)      
            ax = self.proj[method]      

            if len(ax) < 2:
                if hue == None:
                    sns.scatterplot(y=[0]*self.n,x=ax[0], data=self.PlotProj, alpha=0.5)
                    plt.title(titles[i])
                    continue
                elif hue == 'Best':
                    # densiy plot grouped by best algorithm
                    sns.kdeplot(x=ax[0],data=self.PlotProj, hue='Best',fill=True)
                    plt.title(titles[i])
                    continue
                else:
                    # plot ax vs hue
                    sns.scatterplot(x=ax[0],y=hue, data=self.PlotProj, alpha=0.5)
                    plt.title(titles[i])
                    continue


            if i < len(methods)-1:
                sns.scatterplot(x=ax[0],y=ax[1],hue=hue, data=self.PlotProj, alpha=0.5, legend=False)
            
            else:
                sns.scatterplot(x=ax[0],y=ax[1],hue=hue, data=self.PlotProj, alpha=0.5)
                # legend outside plot to right
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                           title=hue if keytitle == '' else keytitle)
        
            plt.title(titles[i])
            plt.xlabel('Z1')
            plt.ylabel('Z2')

        plt.subplots_adjust(wspace=0.25, hspace=0.4)
        if suptitle != '':
            plt.suptitle(suptitle,fontsize='x-large',fontweight='bold')

        plt.show()

        if save != '':
            plt.savefig(save, bbox_inches='tight')


    # adding projections
    def PCA(self, n_components=None):
        """Fit PCA and add 2 dimensional projection to PlotProj. 
            Saves projection matrix with 'all' dimensions in projections dict. 
            Fit on scaled and centered data. 

        Args:
            n_components : number of components to keep. If None, all components are kept.
                If int, specifies the number of components to keep.
                If float, it specifies the cumulative explained variance.
        """

        pca = decomposition.PCA(n_components=n_components, random_state=1)
        pca.fit(self.X_s)

        comps = [f'PC{i+1}' for i in range(pca.n_components_)]
        self.PlotProj = pd.concat(
            [self.PlotProj, 
             pd.DataFrame(pca.transform(self.X_s), columns=comps, index=self.PlotProj.index)], 
             axis=1)

        #self.PlotProj[comps] = pca.transform(self.X_s)
        self.projections['PCA'] = pca
        self.proj['PCA'] = comps
        print("PCA projection added")

    def PILOT(self, analytic=True, ntries=5, scale=False,suffix=''):
        """Fit PILOT and add 2 dimensional projection to PlotProj. 
            To be fit on centered data.

        Args:
            analytic (bool, optional): whether analytic or numerical solver used. Defaults to True.
            ntries (int, optional): Number of runs of numeric solver. Defaults to 5.
        """

        if f'PILOT{suffix}' in self.projections:
            print(f"PILOT{suffix} projection already added")
            return
        
        if scale:
            X,Y = self.X_s, self.Y_s
        else:
            X,Y = self.X_c, self.Y_c

        pilot = pyispace.pilot(X,Y, analytic=analytic, ntries=ntries)
        self.PlotProj[[f'Z{suffix}1',f'Z{suffix}2']] = pilot.Z

        self.projections[f'PILOT{suffix}'] = pilot
        self.proj[f'PILOT{suffix}'] = [f'Z{suffix}1',f'Z{suffix}2']
        print("PILOT projection added")

    def PLSR(self, n_components=None):

        if n_components is None:
            # rank of covariance matrix
            n_components = np.linalg.matrix_rank(np.cov(self.X_s.T))
            name = 'PLSR'
        else:
            name = f'PLSRn{n_components}'

        plsr = cross_decomposition.PLSRegression(n_components=n_components, scale=False)
        plsr.fit(self.X_s, self.Y_s)

        comps = [f'{name}_{i+1}' for i in range(n_components)]
        self.PlotProj = pd.concat(
            [self.PlotProj, 
             pd.DataFrame(plsr.transform(self.X_s), columns=comps, index=self.PlotProj.index)], 
             axis=1)

        self.projections[name] = plsr
        self.proj[name] = comps

        print(f"PLSR projection added with {n_components} dimensions.")

    def PLSC(self, n_components = None):

        if n_components is None:
            n_components = self.a
            name = 'PLSC'
        else:
            name = f'PLSCn{n_components}'

        plsc = cross_decomposition.PLSCanonical(n_components=n_components, scale=False)
        plsc.fit(self.X_s, self.Y_s)

        comps = [f'{name}_{i+1}' for i in range(n_components)]
        self.PlotProj = pd.concat(
            [self.PlotProj, 
             pd.DataFrame(plsc.transform(self.X_s), columns=comps, index=self.PlotProj.index)], 
             axis=1)

        self.projections[name] = plsc
        self.proj[name] = comps

        print(f"PLSC projection added with {n_components} dimensions.")

    def PLSDA(self, n_components = None, method='r'):

        if 'Best' not in self.performance.columns:
            print("No classification data available for PLS DA projection")
            return
        
        # dummy variables for Y labels
        Y_d = pd.get_dummies(self.performance['Best'], dtype=int).values

        if method == 'c':
            n_components = Y_d.shape[1] if n_components is None else n_components
            plsd = cross_decomposition.PLSCanonical(n_components=n_components, scale=False)

        elif method == 'r':
            n_components = np.linalg.matrix_rank(np.cov(self.X_s.T)) if n_components is None else n_components
            plsd = cross_decomposition.PLSRegression(n_components=n_components, scale=False)
                
        
        plsd.fit(self.X_s, Y_d)

        comps = [f'PLSDA{method}{i+1}' for i in range(n_components)]
        self.PlotProj = pd.concat(
            [self.PlotProj, 
             pd.DataFrame(plsd.transform(self.X_s), columns=comps, index=self.PlotProj.index)], 
             axis=1)
        
        self.projections[f'PLSDA{method}'] = plsd
        self.proj[f'PLSDA{method}'] = comps

        print(f"PLS DA ({method}) projection added with {n_components} dimensions.")


    def LDA(self, n_components=None, solver='svd'):

        if 'Best' not in self.performance.columns:
            print("No classification data available for LDA projection")
            return
        
        # need to figure out what to do about 2 class situation 
        lda = LinearDiscriminantAnalysis(solver=solver) #, n_components=n_components)
        pts = lda.fit_transform(self.features.values, self.performance['Best']) 
        dim = min(n_components,pts.shape[1]) if n_components != None else pts.shape[1]

        comps = [f'LD{i+1}' for i in range(dim)]
        self.PlotProj = pd.concat(
            [self.PlotProj, pd.DataFrame(pts, columns=comps, index=self.PlotProj.index)], 
             axis=1)
        # for i in range(dim):
        #     self.PlotProj['LD'+str(i+1)] = pts[:,i]
        
        self.projections['LDA'] = lda
        self.proj['LDA'] = comps
        print(f"LDA projection added in {dim} dimensions.")


    # delete projections
    def delProj(self, method):
        """Deletes projection from PlotProj and projections dict.

        Args:
            method (str): name of the projection to delete
        """
        if method in self.projections.keys():
            del self.projections[method]
            self.PlotProj.drop(columns=self.proj[method], inplace=True)
            del self.proj[method]
            print(f"{method} projection deleted")
        else:
            print(f"{method} projection not defined")    
    
    # projection evaluation
    
    def lda_cv(self, skf, max_it=5000, scale=False, 
                 measures = ['accuracy','precision_w','recall']):
        # if scale - assume data not scaled. scaled here
        # done with training data
        X = self.split_data['X_train']
        Yb = self.split_data['Yb_train']

        n_comps = len(self.performance['Best'].unique()) - 1
        
        cv_scores = list()

        for train, test in skf:
            cv = {'X_train':X[train],'X_test':X[test],
                'Yb_train':Yb.iloc[train],'Yb_test':Yb.iloc[test]}            
        
            if scale:
                scaler = StandardScaler().fit(cv['X_train'])     
                cv['X_train'] = scaler.transform(cv['X_train'])
                cv['X_test'] = scaler.transform(cv['X_test'])   
            
            lda = LinearDiscriminantAnalysis(solver='svd')
            lda.fit(cv['X_train'],cv['Yb_train'])         
                
            comp_Train = lda.transform(cv['X_train'])
            comp_Test = lda.transform(cv['X_test'])    
            
            # fit logistic regression
            for i in range(n_comps):
                lg = LogisticRegression(
                    penalty=None, solver='lbfgs', max_iter=max_it, random_state=111
                    ).fit(comp_Train[:,[n for n in range(i+1)]],cv['Yb_train'])
                
                # predict test and score
                pred = lg.predict(comp_Test[:,[n for n in range(i+1)]])
                scores = {'comps': i+1,
                        'accuracy': accuracy_score(cv['Yb_test'],pred),
                        'precision': precision_score(cv['Yb_test'],pred,average='macro',zero_division=1),
                        'precision_w': precision_score(cv['Yb_test'],pred,average='weighted',zero_division=1),
                        'recall': recall_score(cv['Yb_test'],pred,average='macro',zero_division=1),
                        'f1': f1_score(cv['Yb_test'],pred,average='macro',zero_division=1),
                        'f1_w': f1_score(cv['Yb_test'],pred,average='weighted',zero_division=1)
                        }
                
                cv_scores.append(scores)

        cv_scores = pd.DataFrame(cv_scores).groupby('comps').mean()
        best_d = cv_scores[measures].rank(ascending=False).mean(axis=1).idxmin()
        return {'scores':cv_scores, 'best_d':best_d}

    def pca_cv(self, skf, max_it=5000, scale=False, 
                 measures = ['accuracy','precision_w','recall']):
        # if scale -assume data not scaled. scaled here
        # done with training data
        X = self.split_data['X_train']
        Yb = self.split_data['Yb_train']

        
        cv_scores = list()

        for train, test in skf:
            cv = {'X_train':X[train],'X_test':X[test],
                'Yb_train':Yb.iloc[train],'Yb_test':Yb.iloc[test]}  

            if scale:
                scaler = StandardScaler().fit(cv['X_train'])     
                cv['X_train'] = scaler.transform(cv['X_train'])
                cv['X_test'] = scaler.transform(cv['X_test'])   
            
            pca = decomposition.PCA()
            pca.fit(cv['X_train'])         
            
            n_comps = pca.n_components_
            comp_Train = pca.transform(cv['X_train'])
            comp_Test = pca.transform(cv['X_test'])    
            
            # fit logistic regression
            for i in range(n_comps):
                lg = LogisticRegression(
                    penalty=None, solver='saga', max_iter=max_it, random_state=111
                    ).fit(comp_Train[:,[n for n in range(i+1)]],cv['Yb_train'])
                
                # predict test and score
                pred = lg.predict(comp_Test[:,[n for n in range(i+1)]])
                scores = {'comps': i+1,
                        'accuracy': accuracy_score(cv['Yb_test'],pred),
                        'precision': precision_score(cv['Yb_test'],pred,average='macro',zero_division=1),
                        'precision_w': precision_score(cv['Yb_test'],pred,average='weighted',zero_division=1),
                        'recall': recall_score(cv['Yb_test'],pred,average='macro',zero_division=1),
                        'f1': f1_score(cv['Yb_test'],pred,average='macro',zero_division=1),
                        'f1_w': f1_score(cv['Yb_test'],pred,average='weighted',zero_division=1)
                        }
                
                cv_scores.append(scores)

        cv_scores = pd.DataFrame(cv_scores).groupby('comps').mean()
        best_d = cv_scores[measures].rank(ascending=False).mean(axis=1).idxmin()
        return {'scores':cv_scores, 'best_d':best_d}

    def pilot_cv(self, skf, max_it=5000, scale=False, 
                 measures = ['accuracy','precision_w','recall']):
        # if scale - assume data not scaled. scaled here
        # done with training data
        X = self.split_data['X_train']
        Y = self.split_data['Y_train']
        Yb = self.split_data['Yb_train']

        
        cv_scores = list()

        for train, test in skf:
            cv = {'X_train':X[train],'X_test':X[test],
                'Y_train':Y[train],'Y_test':Y[test],
                'Yb_train':Yb.iloc[train],'Yb_test':Yb.iloc[test]}  

            if scale:
                scalerX = StandardScaler().fit(cv['X_train'])  
                scalerY = StandardScaler().fit(cv['Y_train'])   
                cv['X_train'] = scalerX.transform(cv['X_train'])
                cv['X_test'] = scalerX.transform(cv['X_test']) 
                cv['Y_train'] = scalerY.transform(cv['Y_train'])
                cv['Y_test'] = scalerY.transform(cv['Y_test'])  
            
            pilot = Pilot()
            pilot.fit(cv['X_train'], cv['Y_train'], d=0)

            n_comps = pilot.d_max
            comp_Train = pilot.transform(cv['X_train'])
            comp_Test = pilot.transform(cv['X_test'])    
            
            # fit logistic regression
            for i in range(n_comps):
                lg = LogisticRegression(
                    penalty=None, solver='saga', max_iter=max_it, random_state=111
                    ).fit(comp_Train[:,[n for n in range(i+1)]],cv['Yb_train'])
                
                # predict test and score
                pred = lg.predict(comp_Test[:,[n for n in range(i+1)]])
                scores = {'comps': i+1,
                        'accuracy': accuracy_score(cv['Yb_test'],pred),
                        'precision': precision_score(cv['Yb_test'],pred,average='macro',zero_division=1),
                        'precision_w': precision_score(cv['Yb_test'],pred,average='weighted',zero_division=1),
                        'recall': recall_score(cv['Yb_test'],pred,average='macro',zero_division=1),
                        'f1': f1_score(cv['Yb_test'],pred,average='macro',zero_division=1),
                        'f1_w': f1_score(cv['Yb_test'],pred,average='weighted',zero_division=1)
                        }
                
                cv_scores.append(scores)

        cv_scores = pd.DataFrame(cv_scores).groupby('comps').mean()
        best_d = cv_scores[measures].rank(ascending=False).mean(axis=1).idxmin()
        return {'scores':cv_scores, 'best_d':best_d}

             
             
            
    def fit_and_eval_cv(self, proj_name, K, strat,
                n_comps=None, max_it=5000,plsda_method='r',print_=False):
        """Evaluates a projection method for predicting classes by logistic regression
            using cross validation.

        Args:
            proj_name (str): name of projection method (as defined in proj dict)
            K (int): number of folds for cv
            strat (bool): if stratified cv
            n_comps (int, optional): number of components to fit. Defaults to None.
            max_it (int, optional): max iterations for solver in logistic regression model. Defaults to 5000.
            plsda_method (str, optional): Defaults to 'r'.
            print_ (bool, optional): print proj name before running. Defaults to False.

        Returns:
            cv_scores: average scores of cv
        """
        
        
        if print_:
            print(f'Evaluating {proj_name} {n_comps}')

        if strat:
            kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=1111)
        else:
            kf = KFold(n_splits=K, shuffle=True, random_state=1111)
        
        cv_scores = list()

        for train, test in kf.split(self.X_s,self.performance['Best'].values):
            cv = {'X_train':self.X_s[train],'X_test':self.X_s[test],
                'Y_train':self.Y_s[train],'Y_test':self.Y_s[test],
                'Best_train':self.performance['Best'].values[train],
                'Best_test':self.performance['Best'].values[test]}
            
            if proj_name == 'LDA':
                cv['X_train'] = self.features.values[train]
                cv['X_test'] = self.features.values[test]

                met = LinearDiscriminantAnalysis(n_components=n_comps)
                met.fit(cv['X_train'],cv['Best_train'])
                n_comps = met.explained_variance_ratio_.shape[0]

            elif proj_name == 'PLSC':
                n_comps = self.a if n_comps is None else n_comps
                met = cross_decomposition.PLSCanonical(scale=False,n_components=n_comps)
                met.fit(cv['X_train'],cv['Y_train'])

            elif proj_name == 'PLSR':
                n_comps = np.linalg.matrix_rank(np.cov(self.X_s.T)) if n_comps is None else n_comps
                met = cross_decomposition.PLSRegression(scale=False,n_components=n_comps)
                met.fit(cv['X_train'],cv['Y_train'])


            elif proj_name == 'PLSDA':

                #proj_name = f'PLSDA ({plsda_method})'
                Y_d = pd.get_dummies(cv['Best_train'], dtype=int).values

                if plsda_method == 'c':
                    n_comps = Y_d.shape[1] if n_comps is None else n_comps
                    met = cross_decomposition.PLSCanonical(scale=False,n_components=n_comps)
                elif plsda_method == 'r':
                    n_comps = np.linalg.matrix_rank(np.cov(self.X_s.T)) if n_comps is None else n_comps
                    met = cross_decomposition.PLSRegression(scale=False,n_components=n_comps)
                
                met.fit(cv['X_train'],Y_d)

            elif proj_name == 'PCA':
                met = decomposition.PCA(n_components=n_comps)
                met.fit(cv['X_train'])
                n_comps = met.n_components_
            
            else:
                return
                
            comp_Train = met.transform(cv['X_train'])
            comp_Test = met.transform(cv['X_test'])    
            
            # fit logistic regression
            lg = LogisticRegression(
                penalty=None, solver='lbfgs', max_iter=max_it, random_state=111
                ).fit(comp_Train,cv['Best_train'])
            
            # predict test and score
            pred = lg.predict(comp_Test)
            scores = {'accuracy': accuracy_score(cv['Best_test'],pred),
                    'precision': precision_score(cv['Best_test'],pred,average='macro',zero_division=1),
                    'precision_w': precision_score(cv['Best_test'],pred,average='weighted',zero_division=1),
                    'recall': recall_score(cv['Best_test'],pred,average='macro',zero_division=1),
                    'f1': f1_score(cv['Best_test'],pred,average='macro',zero_division=1),
                    'f1_w': f1_score(cv['Best_test'],pred,average='weighted',zero_division=1)
                    }
            
            cv_scores.append(scores)

        cv_scores = dict(pd.DataFrame(cv_scores).mean())
        cv_scores['name'] = proj_name 
        cv_scores['n_comps'] = n_comps


        return cv_scores

    def get_components_cv(self, proj_name, K, strat,n_max=0,
                plsda_method='r', max_it=5000, print_=False,
                measures=['accuracy','precision','precision_w','recall'],
                show_plot=False, return_scores=False):

        if n_max > 0:
            pass
        elif proj_name in ['PLSR','PLSDA']:
            n_max = np.linalg.matrix_rank(np.cov(self.X_s.T))
            # PCA included because of quirk when doing CV
            
        elif proj_name == 'PCA':
            n_max = decomposition.PCA().fit(self.X_s).n_components_
        else:
            return
        
        scores = pd.DataFrame(
            [self.fit_and_eval_cv(proj_name, K, n_comps=i, 
                                  plsda_method=plsda_method, strat=strat,
                                  max_it=max_it,print_=print_) for i in range(1,n_max+1)]
        )
        scores.set_index('n_comps',inplace=True)

        if show_plot:
            scores[measures].plot()
            plt.show()
        
        if return_scores:
            return scores
        
        n_best = scores[measures].rank(ascending=False).mean(axis=1).idxmin()
        by_measure = scores[measures].idxmax(axis=0)

        return n_best, by_measure
        


    def eval_projections(self, proj_name, K, dim=0, w=None):

        proj = self.PlotProj[self.proj[proj_name]]
        if dim > 0:
            proj = proj.iloc[:,:dim]

        D_high = distance.squareform(distance.pdist(self.X_s, 'euclidean'))
        D_low = distance.squareform(distance.pdist(proj, 'euclidean'))

        metrics = {
            'intrinsic_dim': self.intrinsic_dim_ratio(),
            'normalised_stress': self.normalised_stress(proj, (D_high, D_low)),
            'shepard_goodness': self.shepard_goodness(proj, (D_high, D_low))
        }

        if w != None:
            metrics['normalised_stress_weighted'] = self.normalised_stress_weighted(proj, w, (D_high, D_low))

        for k in K:
            metrics[f'trustworthiness_k{k}'] = self.trustworthiness(proj, k, (D_high, D_low))
            metrics[f'continuity_k{k}'] = self.continuity(proj, k, (D_high, D_low))
            metrics[f'neighbourhood_hit_k{k}'] = self.neighbourhood_hit(proj, k, dist=(D_high, D_low))
            metrics[f'neighbourhood_hit_rel_k{k}'] = self.neighbourhood_hit(proj, k, rel=True, dist=(D_high, D_low))

        return metrics
        

    
    def intrinsic_dim_ratio(self):

        pca = decomposition.PCA()
        pca.fit(self.X_s)

        intrinsic_dim = np.where(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1

        return intrinsic_dim / pca.n_components_
    
    def trustworthiness(self, proj, k, dist=None):

        if dist == None:
        # distance matrices of original and projected data
            D_high = distance.squareform(distance.pdist(self.X_s, 'euclidean'))
            D_low = distance.squareform(distance.pdist(proj, 'euclidean'))
        else:
            D_high, D_low = dist

        # nearest neighbors
        nn_orig = D_high.argsort()
        nn_proj = D_low.argsort()

        knn_orig = nn_orig[:, 1:k + 1]
        knn_proj = nn_proj[:, 1:k + 1]

        sum_T = 0
        
        for i in range(self.n):
            # NN in proj space that are not in orig space
            U = np.setdiff1d(knn_proj[i], knn_orig[i])
            u_ranks = [(nn_orig[i].tolist().index(u) - k) for u in U]       
            sum_T += sum(u_ranks)
        
        mult = 2 / (self.n*k * (2*self.n - 3*k - 1))

        return 1 - mult * sum_T

    def continuity(self, proj, k, dist=None):

        if dist == None:
        # distance matrices of original and projected data
            D_high = distance.squareform(distance.pdist(self.X_s, 'euclidean'))
            D_low = distance.squareform(distance.pdist(proj, 'euclidean'))
        else:
            D_high, D_low = dist

        # nearest neighbors
        nn_orig = D_high.argsort()
        nn_proj = D_low.argsort()

        knn_orig = nn_orig[:, 1:k + 1]
        knn_proj = nn_proj[:, 1:k + 1]

        sum_C = 0

        for i in range(self.n):        
            # NN in orig space that are not in proj space
            V = np.setdiff1d(knn_orig[i], knn_proj[i])
            v_ranks = [(nn_proj[i].tolist().index(v) - k) for v in V]
            sum_C += sum(v_ranks)

        mult = 2 / (self.n*k * (2*self.n - 3*k - 1))
        return 1 - mult * sum_C

    def normalised_stress(self, proj, dist=None):

        if dist == None:
        # distance matrices of original and projected data
            D_high = distance.squareform(distance.pdist(self.X_s, 'euclidean'))
            D_low = distance.squareform(distance.pdist(proj, 'euclidean'))
        else:
            D_high, D_low = dist

        # normalised stress
        return np.sum((D_high - D_low)**2) / np.sum(D_high**2)
    
    def normalised_stress_weighted(self, proj, w, dist=None):

        # TODO - figure a good w. should be a matrix

        if dist == None:
        # distance matrices of original and projected data
            D_high = distance.squareform(distance.pdist(self.X_s, 'euclidean'))
            D_low = distance.squareform(distance.pdist(proj, 'euclidean'))
        else:
            D_high, D_low = dist

        # normalised stress
        return np.sum(w * (D_high - D_low)**2) / np.sum(w * D_high**2)
    
    def neighbourhood_hit(self, proj, k, rel=False, dist=None):

        if 'Best' not in self.performance.columns:
            print("No classification data available for evaluation")
            return None
        
        y = self.performance['Best'].values

        if type(proj) == str:
            proj = self.PlotProj[self.proj[proj]]

        if dist == None:
        # distance matrices of original and projected data
            D_high = distance.squareform(distance.pdist(self.X_s, 'euclidean'))
            D_low = distance.squareform(distance.pdist(proj, 'euclidean'))
        else:
            D_high, D_low = dist
        
        nn_proj = D_low.argsort()
        knn_proj = nn_proj[:, 1:k + 1]

        hits_proj = [np.mean(y[knn_proj[i]] == y[i]) for i in range(self.n)]

        if rel:
            nn_orig = D_high.argsort()
            knn_orig = nn_orig[:, 1:k + 1]

            hits_orig = [np.mean(y[knn_orig[i]] == y[i]) for i in range(self.n)]

            return np.mean(hits_proj) / np.mean(hits_orig)
        

        return np.mean(hits_proj)
    
    def shepard_goodness(self, proj, dist=None):
        
        if dist == None:
        # distance matrices of original and projected data
            D_high = distance.squareform(distance.pdist(self.X_s, 'euclidean'))
            D_low = distance.squareform(distance.pdist(proj, 'euclidean'))
        else:
            D_high, D_low = dist
            
        # goodness
        return stats.spearmanr(D_high, D_low, axis=None)[0]

            
def matildaTSP_IS():
    matildaTSP = pd.read_csv('./TSP data/matilda metadata.csv',index_col=0)
    
    TSP1 = InstanceSpace()
    TSP1.fromMetadata(matildaTSP)
    
    bestExp = lambda x: 'ANY' if abs(x.algo_CLK_Mean_Effort - x.algo_LKCC_Mean_Effort) < 1e-3 else ('CLK' if x.algo_CLK_Mean_Effort < x.algo_LKCC_Mean_Effort else 'LKCC')
    TSP1.getBest(bestExp,axis=1)
    
    #TSP1.initPlotData('best')

    # Projections
    TSP1.PCA()
    TSP1.PILOT(suffix='_a')
    TSP1.PILOT(analytic=False,ntries=5,suffix='_n')

    # Plots
    TSP1.plots2D(['PCA','PILOT_a','PILOT_n'],(1,3))
    return TSP1

def data_IS(datapath, prefixes=['feature_','algo_'],best=None):
    metadata = pd.read_csv(datapath,index_col=0)
    
    IS1 = InstanceSpace()
    IS1.fromMetadata(metadata,prefixes)
    
    if best != None:
        IS1.getBest(best,axis=1)
        #IS1.initPlotData('best')
    
    # Projections
    IS1.PCA()
    IS1.PILOT(suffix='_a')
    IS1.PILOT(analytic=False,ntries=5,suffix='_n')

    # Plots
    IS1.plots2D(['PCA','PILOT_a','PILOT_n'],(1,3))
    return IS1



matildaTSP = pd.read_csv('./TSP data/matilda metadata.csv',index_col=0)
TSP_abs = InstanceSpace()
TSP_abs.fromMetadata(matildaTSP)

X = TSP_abs.X_s
Y = TSP_abs.Y_s

# PLS 
def pls(X,Y,d):

    r = 0
    Xs = list()
    Ys = list()
    us = list()
    vs = list()
    Gammas = list()
    Tildes = list()
    Omegas = list()
    Xhats = list()
    Yhats = list()

    # step 2
    Xs.append(np.mat(X))
    Ys.append(np.mat(Y))

    while r < d or (np.abs(Xs[r].T @ Ys[r]) > 10e-4).any():
        # step 3
        U,_,V = np.linalg.svd(Xs[r].T @ Ys[r])
        us.append(U[0].T) # first column of U
        vs.append(V[0].T)

        # step 4
        Tildes.append(Xs[r] @ us[r])
        Omegas.append(Ys[r] @ vs[r])

        # step 5
        Gammas.append((Tildes[r].T @ Xs[r])/(Tildes[r].T @ Tildes[r])[0,0])
        Xhats.append(Tildes[r] @ Gammas[r])
        Yhats.append((Omegas[r] @ Omegas[r].T @ Ys[r])/(Omegas[r].T @ Omegas[r])[0,0])

        # step 6
        Xs.append(Xs[r] - Xhats[r])
        Ys.append(Ys[r] - Yhats[r])

        # step 7
        r += 1

    # us to 2d array
    Us = np.array(us).reshape(len(us),-1).T
    Gammas = np.array(Gammas).reshape(len(Gammas),-1) #already T
    
    # P - projection matrix (rotation of X)
    P = Us @ np.linalg.inv(Gammas @ Us)

    return P

if __name__ == "__main__":
    
   pass
   
    
    