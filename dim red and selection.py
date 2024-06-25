import IS_class as ip

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn import cross_decomposition
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, make_scorer, confusion_matrix
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns

import pickle as pkl

def transform_and_predict(metadata, prefixes, best, obj_min, test_size, cv_k, rand, 
                          folder_out, max_it=5000, skip_proj=[]):

    Preds = {}
    Models = {}
    Projs = {}
    N_comps = {}

    do_proj = {k:True if k not in skip_proj else False 
               for k in ['All','Naive','PLS','PCA','LDA','PILOT']}

    # 1. load meta data
    IP = ip.InstanceSpace()
    IP.fromMetadata(metadata, prefixes)

    # 2. get best - array created outside of function
    # .. split train test
    IP.getBest(best)
    IP.getRelativePerf(obj_min)
    IP.splitData2(test_size, rand) # using scaled features for all models

    train_ind = list(IP.split_data['Yb_train'].index)
    test_ind = list(IP.split_data['Yb_test'].index)

    Preds[('actual','train')] = IP.split_data['Yb_train']
    Preds[('actual','test')] = IP.split_data['Yb_test']

    scalerX = StandardScaler().fit(IP.split_data['X_train'])
    scalerY = StandardScaler().fit(IP.split_data['Y_train'])
    Xs = {t:scalerX.transform(IP.split_data[f'X_{t}']) for t in ['train','test']}
    Ys = {t:scalerY.transform(IP.split_data[f'Y_{t}']) for t in ['train','test']}

    # cv splits
    skf = [t for t in StratifiedKFold(
        n_splits=cv_k, shuffle=True, random_state=111).split(
        IP.split_data['X_train'], IP.split_data['Yb_train'])]
    
    measures=['accuracy','precision_w','recall']


    # 3. DUMMY prediction - eval metrics AND predicitons
    if do_proj['Naive']:
        print('fitting dummy')
        dc = DummyClassifier(strategy='most_frequent')
        dc.fit(IP.split_data['X_train'], IP.split_data['Yb_train'])

        Preds[('Naive','train')] = dc.predict(IP.split_data['X_train'])
        Preds[('Naive','test')] = dc.predict(IP.split_data['X_test'])
        Models['Naive'] = dc
        N_comps['Naive'] = 0
        

    # 4. log reg all feats
    if do_proj['All']:
        print('fitting log reg with all feats')
        lg = LogisticRegression(penalty=None,solver='saga',max_iter=max_it, random_state=rand)
        lg.fit(Xs['train'],IP.split_data['Yb_train'])

        Preds[('All','train')] = lg.predict(Xs['train'])
        Preds[('All','test')] = lg.predict(Xs['test'])
        Models['All'] = lg
        N_comps['All'] = len([c for c in lg.coef_[0] if c != 0])


    def proj_and_fit(md, name, one):
        Models[name] = md
        
        projTrain = md.transform(Xs['train'])
        projTest = md.transform(Xs['test'])         
        
        print(f'Fitting log reg with {name} comps')
        lg = LogisticRegression(penalty=None,solver='saga',max_iter=max_it, random_state=rand)
        if one:
            lg.fit(projTrain[:,0].reshape(-1, 1), IP.split_data['Yb_train'])
            predTrain = lg.predict(projTrain[:,0].reshape(-1, 1))
            predTest = lg.predict(projTest[:,0].reshape(-1, 1))
        else:
            lg.fit(projTrain, IP.split_data['Yb_train'])
            predTrain = lg.predict(projTrain)
            predTest = lg.predict(projTest)

        projTrain = pd.DataFrame(projTrain, 
                columns=[f'{name}_{i}' for i in range(projTrain.shape[1])], 
                index=train_ind)
        projTrain['Group'] = 'train'
        projTrain['Actual'] = IP.split_data['Yb_train']
        projTrain['Pred'] = predTrain

        projTest = pd.DataFrame(projTest,
                columns=[f'{name}_{i}' for i in range(projTest.shape[1])],
                index=test_ind)
        projTest['Group'] = 'test'
        projTest['Actual'] = IP.split_data['Yb_test']
        projTest['Pred'] = predTest

        Projs[name] = pd.concat([projTrain, projTest])                
        Preds[(name,'train')] = predTrain
        Preds[(name,'test')] = predTest
        Models[f'lg {name}'] = lg

    
    if do_proj['PLS']:
        # 5. CV to determine comps for PLS
        max_pls_comp = min(len(IP.split_data['X_train']),
            np.linalg.matrix_rank(IP.split_data['X_train']))

        # name, y, scale
        plsSets = [('PLS',Ys['train'],False),
                ('rel PLS',IP.split_data['Yr_train'],True), 
                ('PLS DA',pd.get_dummies(IP.split_data['Yb_train'], dtype=float),False)]
        
        for mod, y, s in plsSets:

            # projection
            pls = GridSearchCV(
                cross_decomposition.PLSRegression(scale=s),
                param_grid={'n_components':range(1,max_pls_comp)},
                scoring='neg_mean_squared_error',
                cv=skf,
                return_train_score=True
            )
            pls.fit(Xs['train'], y)
            # print(pls.best_params_)

            N_comps[mod] = pls.best_params_['n_components']
            
            if pls.best_params_['n_components'] == 1:
                print(f'WARNING: only 1 comp for {mod}')
                pls2 = cross_decomposition.PLSRegression(n_components=2,scale=s)
                pls2.fit(Xs['train'], y)
                proj_and_fit(pls2, mod, True)
            else:
                proj_and_fit(pls.best_estimator_, mod, False)
            
        
    if do_proj['LDA']:
    # 7. LDA - 5 and 6    
        print('fitting LDA')
        ldaCV = IP.lda_cv(skf, max_it, True)
        N_comps['LDA'] = ldaCV['best_d']
        
        lda = LinearDiscriminantAnalysis(n_components=N_comps['LDA'])
        lda.fit(Xs['train'], IP.split_data['Yb_train'])  

        if N_comps['LDA'] == 1 and len(IP.performance['Best'].unique())>2:
            print('WARNING: only 1 comp for LDA')
            lda2 = LinearDiscriminantAnalysis(n_components=2)
            lda2.fit(Xs['train'], IP.split_data['Yb_train'])
            proj_and_fit(lda2, 'LDA',True)
        else:
            proj_and_fit(lda, 'LDA', False)
        
    if do_proj['PCA']:
    # 8. PCA - 5 and 6
        print('fitting PCA')
        pcaCV = IP.pca_cv(skf, max_it, True)
        N_comps['PCA'] = pcaCV['best_d']

        pca = decomposition.PCA(n_components=N_comps['PCA'])
        pca.fit(Xs['train'])

        if N_comps['PCA'] == 1:
            print('WARNING: only 1 comp for PCA')
            pca2 = decomposition.PCA(n_components=2)
            pca2.fit(Xs['train'])
            proj_and_fit(pca2, 'PCA', True)
        else:
            proj_and_fit(pca, 'PCA', False)


    # 9. Pilot - 5 and 6
    if do_proj['PILOT']:
        print('fitting PILOT')
        pilotCV = IP.pilot_cv(skf, max_it, True)
        N_comps['PILOT'] =  pilotCV['best_d']

        pilot = ip.Pilot()
        pilot.fit(Xs['train'],Ys['train'],N_comps['PILOT'])
        
        if N_comps['PILOT'] == 1:
            print('WARNING: only 1 comp for PILOT')
            pilot2 = ip.Pilot()
            pilot2.fit(Xs['train'],Ys['train'],2)
            proj_and_fit(pilot2, 'PILOT', True)
        else:
            proj_and_fit(pilot, 'PILOT', False)
        
    
    
    
    Models['n_comps'] = N_comps

    # save models to pkl
    with open(f'{folder_out}/models.pkl', 'wb') as f:
        pkl.dump(Models, f)
    with open(f'{folder_out}/predictions.pkl', 'wb') as f:
        pkl.dump(Preds, f)
    with open(f'{folder_out}/projections.pkl', 'wb') as f:
        pkl.dump(Projs, f)
    with open(f'{folder_out}/data.pkl', 'wb') as f:
        pkl.dump(IP, f)

    return {'preds':Preds,'proj': Projs, 'data': IP, 'n_comps': N_comps}

    
# evaluation metrics
metrics = {'accuracy':make_scorer(accuracy_score),
    'precision': make_scorer(precision_score,zero_division=1,average='macro'),
    'precision_w': make_scorer(precision_score,zero_division=1,average='weighted'),    
    'recall':make_scorer(recall_score,zero_division=1,average='macro'),
    'f1':make_scorer(f1_score,zero_division=1,average='macro'),
    'f1_w':make_scorer(f1_score,zero_division=1,average='weighted'),
    }

def class_metrics(y_true, y_pred, classes):
    metrics = {'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=1,average='macro'),
            'precision_w': precision_score(y_true, y_pred, zero_division=1,average='weighted'),
            'recall': recall_score(y_true, y_pred, zero_division=1,average='macro'),
            'f1': f1_score(y_true, y_pred, zero_division=1,average='macro'),
            'f1_w': f1_score(y_true, y_pred, zero_division=1,average='weighted'),
            'conf': confusion_matrix(y_true, y_pred, labels=classes)
            }

    return metrics


def eval_predictions(data, preds, classes, n_comps, folder_out):

    trainMetrics = pd.DataFrame({mod: class_metrics(
        data['Yb_train'], preds[(mod,t)], classes) 
        for (mod,t) in preds.keys() if t == 'train'
    }).T
    trainMetrics['n_comps'] = n_comps

    testMetrics = pd.DataFrame({mod: class_metrics(
        data['Yb_test'], preds[(mod,t)], classes) 
        for (mod,t) in preds.keys() if t == 'test'
    }).T
    testMetrics['n_comps'] = n_comps

    trainMetrics.to_csv(f'{folder_out}/train_metrics.csv')
    testMetrics.to_csv(f'{folder_out}/test_metrics.csv')

    return trainMetrics, testMetrics

def genResultPlots(proj, out, hue_order=None,fgsize=(12,8),alpha=0.5,
                   methods = ['PCA','PILOT','LDA','PLS','rel PLS','PLS DA']):    

    # actual label plots
    plt.figure(figsize=fgsize)
    for i,mod in enumerate(methods):
        plt.subplot(2,3,i+1)
        sns.scatterplot(data=proj[mod], x=f'{mod}_0', y=f'{mod}_1', 
                        hue='Actual', style='Group', alpha=alpha,
                        hue_order=hue_order,
                        legend="auto" if i == len(methods)-1 else False)
        plt.title(mod)
        plt.xlabel('Comp 1')
        plt.ylabel('Comp 2')

        if i == len(methods)-1:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f'./conf paper plots/{out} act class.pdf', bbox_inches='tight')

    # pred label plots
    plt.figure(figsize=fgsize)
    for i,mod in enumerate(methods):
        plt.subplot(2,3,i+1)
        sns.scatterplot(data=proj[mod], x=f'{mod}_0', y=f'{mod}_1', 
                        hue='Pred', style='Group', alpha=alpha,
                        hue_order=hue_order,
                        legend="auto" if i == len(methods)-1 else False)
        plt.title(mod)
        plt.xlabel('Comp 1')
        plt.ylabel('Comp 2')

        if i == len(methods)-1:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f'./conf paper plots/{out} pred class.pdf', bbox_inches='tight')

def genRegret(Ip, folder,
              methods = ['Naive','All','PCA','PILOT','LDA','PLS','rel PLS','PLS DA']):
    
    
    ind = {t: list(Ip.split_data[f'Yb_{t}'].index) for t in ['train','test']}

    preds = pd.read_pickle(f'{folder}/predictions.pkl')
    preds = {t: pd.DataFrame(
        {mod: preds[(mod,t_)] for (mod,t_) in preds.keys() if t_==t}, index=ind[t])
    for t in ['train','test']}
    
    cols = [a.split('_')[1] for a in Ip.algorithms]
    Yr = {t: pd.DataFrame(Ip.split_data[f'Yr_{t}'], columns = cols, index = ind[t]) for t in ['train','test']}
    
    regrets = {t: 
        pd.DataFrame({mod:
            [Yr[t].loc[inst, preds[t].loc[inst,mod]] for inst in ind[t]]
        for mod in methods}, index=ind[t]) 
    for t in ['train','test']}

    regrets_sub = {t: 
        pd.DataFrame({mod:
            [Yr[t].loc[inst, preds[t].loc[inst,mod]]
             if preds[t].loc[inst,mod] != Ip.split_data[f'Yb_{t}'].loc[inst] else np.nan
              for inst in ind[t]]
        for mod in methods}, index=ind[t]) 
    for t in ['train','test']}

    return regrets, regrets_sub

def genRegret_abs(Ip,folder, min=True,
                  methods = ['Naive','All','PCA','PILOT','LDA','PLS','rel PLS','PLS DA']):
    
    if min:
        Ydiff = Ip.performance[Ip.algorithms].apply(
            lambda row: row - row.min(), axis=1
        ).fillna(0)#.values
    else:
        Ydiff = Ip.performance[Ip.algorithms].apply(
            lambda row: row - row.max(), axis=1
        ).fillna(0)#.values

    

    ind = {t: list(Ip.split_data[f'Yb_{t}'].index) for t in ['train','test']}

    preds = pd.read_pickle(f'{folder}/predictions.pkl')
    preds = {t: pd.DataFrame(
        {mod: preds[(mod,t_)] for (mod,t_) in preds.keys() if t_==t}, index=ind[t])
    for t in ['train','test']}
    
    #cols = [a.split('_')[1] for a in Ip.algorithms]
    Yr = {t: Ydiff.loc[ind[t],:].rename(columns=lambda a:a.split('_')[1]) 
          for t in ['train','test']}
    
    regrets = {t: 
        pd.DataFrame({mod:
            [Yr[t].loc[inst, preds[t].loc[inst,mod]] for inst in ind[t]]
        for mod in methods}, index=ind[t]) 
    for t in ['train','test']}

    regrets_sub = {t: 
        pd.DataFrame({mod:
            [Yr[t].loc[inst, preds[t].loc[inst,mod]]
             if preds[t].loc[inst,mod] != Ip.split_data[f'Yb_{t}'].loc[inst] else np.nan
              for inst in ind[t]]
        for mod in methods}, index=ind[t]) 
    for t in ['train','test']}

    return regrets, regrets_sub
    
def neighbourhood_hit(proj, name, comps, label, k):

    y = proj[label].values

    
    # distance matrices of original and projected data
    D_low = distance.squareform(
        distance.pdist(proj[[f'{name}_{i}' for i in range(comps)]], 'euclidean'))

    nn_proj = D_low.argsort()
    knn_proj = nn_proj[:, 1:k + 1]

    hits_proj = [np.mean(y[knn_proj[i]] == y[i]) for i in range(proj.shape[0])]

    

    return np.mean(hits_proj)


def nhRates(proj, k_max):
    methods = proj.keys()
    nh = {}

    nh['Actual'] = pd.DataFrame(
        {mod: [neighbourhood_hit(proj[mod],mod,2,'Actual', k) for k in range(1,k_max+1)] 
            for mod in methods}, index=range(1,k_max+1))
    
    nh['Predicted'] = pd.DataFrame(
        {mod: [neighbourhood_hit(proj[mod],mod,2,'Pred', k) for k in range(1,k_max+1)] 
            for mod in methods}, index=range(1,k_max+1))
    
    return nh
    

##### MAIN #####
tsp60data = pd.read_csv('./Data/metaTSP60.csv',index_col=0)
tsp1800data = pd.read_csv('./Data/metaTSP1800.csv',index_col=0)
tspWdata = pd.read_csv('./Data/metaTSPwinner.csv',index_col=0)

matildaTSP = pd.read_csv('./Data/matilda metadata.csv',index_col=0)
bestExpM = lambda x: 'CLK' if x.algo_CLK_Mean_Effort < x.algo_LKCC_Mean_Effort else 'LKCC'

knapsack = pd.read_csv('./Data/knapsack metadata.csv', index_col=0)
bestKP = pd.read_csv('./Data/knapsack_best.csv', index_col=0)

vrpData = pd.read_csv('./Data/setX meta_abs.csv', index_col=0)
bestVRP = pd.read_csv('./Data/setX best.csv', index_col=0)

### knapsack
kp = transform_and_predict(
    metadata = knapsack, prefixes=['feature_','algo_'],
    best=bestKP, obj_min=True,
    test_size=0.2, cv_k=5, rand=1111, max_it=8000,
    folder_out='./Proj Results/knapsack'
)

kpEval = eval_predictions(data=kp['data'].split_data, preds=kp['preds'],
                classes=bestKP['Best'].unique(), n_comps=kp['n_comps'],
                folder_out='./Proj Results/knapsack')

genResultPlots(kp['proj'], 'knapsack', 
               hue_order=['Minknap', 'Expknap','Combo'], fgsize=(12,8), alpha=0.4)

kpOther = {'reg abs':genRegret_abs(kp['data'],'./Proj Results/knapsack', min=True),
         'reg rel':genRegret(kp['data'],'./Proj Results/knapsack'),
         'nh':nhRates(kp['proj'], 40)}

with open('./Proj Results/knapsack/other.pkl', 'wb') as f:
    pkl.dump(kpOther, f)

### vrp
vrpData = vrpData.drop(columns=['perf_ortools', 'perf_LKH3', 'perf_KGLS'])
vrp = transform_and_predict(
    metadata = vrpData, prefixes=['feat_','perf_'],
    best=bestVRP['Best'], obj_min=True,
    test_size=0.2, cv_k=5, rand=1111, max_it=8000,
    folder_out='./Proj Results/vrp', skip_proj=['PILOT']
)

vrpEval = eval_predictions(data=vrp['data'].split_data, 
                preds=vrp['preds'],
                classes=bestVRP['Best'].unique(), n_comps=vrp['n_comps'],
                folder_out='./Proj Results/vrp')

genResultPlots(vrp['proj'], 'vrp',
            hue_order=['HGS-CVRP','FILO', 'SISR', 'HILS', 'HGS-2012'], 
            fgsize=(12,8), alpha=0.6,
            methods = ['PCA','LDA','PLS','rel PLS','PLS DA'])

methodsV = ['Naive','All','PCA','LDA','PLS','rel PLS','PLS DA']
vrpOther = {'reg abs':genRegret_abs(vrp['data'],'./Proj Results/vrp', min=True, methods=methodsV),
            'reg rel':genRegret(vrp['data'],'./Proj Results/vrp', methods=methodsV),
            'nh':nhRates(vrp['proj'], 10)}

with open('./Proj Results/vrp/other.pkl', 'wb') as f:
    pkl.dump(vrpOther, f)


### tsp60
tsp60 = transform_and_predict(
    metadata = tsp60data, prefixes=['feat_','perf_'],
    best=tsp60data['CLASS'], obj_min=True,
    test_size=0.2, cv_k=5, rand=1111, max_it=8000,
    folder_out='./Proj Results/tsp60'
)

tsp60Eval = eval_predictions(data=tsp60['data'].split_data, 
                preds=tsp60['preds'],
                classes=tsp60data['CLASS'].unique(), n_comps=tsp60['n_comps'],
                folder_out='./Proj Results/tsp60')

genResultPlots(tsp60['proj'], 'tsp60',
            hue_order=['LKH','MAOS','ANY'], fgsize=(12,8), alpha=0.6)

tsp60Other = {'nh':nhRates(tsp60['proj'], 25)}
with open('./Proj Results/tsp60/other.pkl', 'wb') as f:
    pkl.dump(tsp60Other, f)

# ### tsp1800
tsp1800 = transform_and_predict(
    metadata = tsp1800data, prefixes=['feat_','perf_'],
    best=tsp1800data['CLASS'], obj_min=True,
    test_size=0.2, cv_k=5, rand=1111, max_it=8000,
    folder_out='./Proj Results/tsp1800'
)
tsp1800Eval = eval_predictions(data=tsp1800['data'].split_data,
                preds=tsp1800['preds'],
                classes=tsp1800data['CLASS'].unique(), n_comps=tsp1800['n_comps'],
                folder_out='./Proj Results/tsp1800')

genResultPlots(tsp1800['proj'], 'tsp1800',
            hue_order=['LKH','MAOS','ANY'], fgsize=(12,8), alpha=0.6)

tsp1800Other = {'nh':nhRates(tsp1800['proj'], 25)}
with open('./Proj Results/tsp1800/other.pkl', 'wb') as f:
    pkl.dump(tsp1800Other, f)

### tspW
tspW = transform_and_predict(
    metadata = tspWdata, prefixes=['feat_','perf_'],
    best=tspWdata['CLASS'], obj_min=True,
    test_size=0.2, cv_k=5, rand=1111, max_it=8000,
    folder_out='./Proj Results/tspWin'
)
tspWEval = eval_predictions(data=tspW['data'].split_data,
                preds=tspW['preds'],
                classes=tspWdata['CLASS'].unique(), n_comps=tspW['n_comps'],
                folder_out='./Proj Results/tspWin')

genResultPlots(tspW['proj'], 'tspW',
            hue_order=['LKH','MAOS','ANY'], fgsize=(12,8), alpha=0.6)

tspWOther = {'nh':nhRates(tspW['proj'], 25)}
with open('./Proj Results/tspWin/other.pkl', 'wb') as f:
    pkl.dump(tspWOther, f)