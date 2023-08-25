import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from operon_sklearn import SymbolicRegressor
from pyoperon import InfixFormatter, FitLeastSquares, MSE
import csv
import argparse

def operon_cmb(y,X,test_name,cal_individual):
    '''
    # y: the encoder outputs,[batch,6]
    # X: the cosmological parameters,[batch,6]
    # test_name: a specific name for this test
    # cal_individual: whether to calculate the statistics related to the individual expressions (might encounter NaN bug)
    '''
    # Do test-train split
    X_train = X.copy()
    X_test = X.copy()
    y_train = y.copy()
    y_test = y.copy()
    
    # Make SR class
    reg = SymbolicRegressor(
            #allowed_symbols='add,mul,pow,constant,variable', # cause the NaN error later for individual expressions
            allowed_symbols='add,sub,mul,aq,sin,constant,variable',
            offspring_generator='basic',
            local_iterations=1000,
            max_length=50,
            initialization_method='btc',
            n_threads=32,
            objectives = ['r2', 'length'],
            epsilon = 1e-3,
            random_state=None,
            reinserter='keep-best',
            max_evaluations=int(1e5),
            symbolic_mode=False
            )
    
    # Run fit
    reg.fit(X_train, y_train)
    
    compute_mse = MSE()
    
    # Output pareto front to file 
    print('Outputting pareto')
    with open('./data/sr/pareto_%s.csv'%test_name, 'w') as f:
        writer = csv.writer(f, delimiter=',')
    
        writer.writerow(['length',
                        'r2_train',
                        'r2_test',
                        'mse_train',
                        'mse_test',
                        'scale',
                        'offset',
                        'infix',
        ])
    
        count = 0
        pred_ind = np.zeros((len(reg.pareto_front_),500))
        pareto_mse = np.zeros((len(reg.pareto_front_)))
        for model, model_vars, model_obj, bic in reg.pareto_front_:
            y_pred_train = reg.evaluate_model(model, X_train)
            y_pred_test = reg.evaluate_model(model, X_test)
    
            scale, offset = FitLeastSquares(y_pred_train, y_train)
            y_pred_train = scale * y_pred_train + offset
            y_pred_test = scale * y_pred_test + offset
            
            pred_ind[count] = y_pred_test
            mse = np.mean((y_train- y_pred_train)**2)
            pareto_mse[count] = mse
            #np.save('./data/pred_individuals5_%d.npy'%count,y_pred_test)
            count += 1
    
            variables = { v.Hash : v.Name for v in model_vars }
            writer.writerow([model.Length,
                            r2_score(y_train, y_pred_train),
                            r2_score(y_test, y_pred_test),
                            compute_mse(y_train, y_pred_train),
                            compute_mse(y_test, y_pred_test),
                            scale,
                            offset,
                            InfixFormatter.Format(model, variables, 3),
            ])
        np.save('./data/sr/pred_pareto_%s.npy'%test_name,pred_ind)
        np.save('./data/sr/mse_pareto_%s.npy'%test_name,pareto_mse)
        
    if cal_individual==True:
        # Output currently considered individuals to file
        print(f'Outputting {len(reg.individuals_)} individuals')
        with open('./data/sr/individuals_%s.csv'%test_name, 'w') as f:
            writer = csv.writer(f, delimiter=',')
        
            writer.writerow(['length',
                            'r2_train',
                            'r2_test',
                            'mse_train',
                            'mse_test',
                            'scale',
                            'offset',
                            'infix',
            ])
        
            count = 0
            pred_ind = np.zeros((2000,500))
            for ind in reg.individuals_:
                y_pred_train = reg.evaluate_model(ind.Genotype, X_train)
                y_pred_test = reg.evaluate_model(ind.Genotype, X_test)
        
                scale, offset = FitLeastSquares(y_pred_train, y_train)
                y_pred_train = scale * y_pred_train + offset
                y_pred_test = scale * y_pred_test + offset
        
                pred_ind[count] = y_pred_test
                #np.save('./data/pred_individuals5_%d.npy'%count,y_pred_test)
                count += 1
                
                #print (y_pred_train)
                writer.writerow([ind.Genotype.Length,
                                r2_score(y_train, y_pred_train),
                                r2_score(y_test, y_pred_test),
                                compute_mse(y_train, y_pred_train),
                                compute_mse(y_test, y_pred_test),
                                scale,
                                offset,
                                reg.get_model_string(ind.Genotype, 4)
                ])
            np.save('./data/sr/pred_individuals_%s.npy'%test_name,pred_ind)
            
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        help="The basic name of encoder-decoder experiments",
                        type=str,
                        default='shallow96-test',
                        required=False)
    
    parser.add_argument("--operon_name",
                        help="The basic name of the operon experiments",
                        type=str,
                        default='shallow96-test',
                        required=False)
    parser.add_argument("--cmb_type",
                        help="TT,TE,or EE",
                        type=str,
                        default='TT',
                        required=False)
    
    args = parser.parse_args()
    ############################################################### Key parameters to vary
    cmb_type=args.cmb_type # CMB type
    name=cmb_type+'_'+args.model_name # The basic name of this experiments
    operon_name = cmb_type+'_'+args.operon_name # The basic name for the operon
    d_encode=6 #128, the bottleneck dimension
    ################################################################
    
    n_latent = d_encode
    y=np.load('./data/camb_new_processed/encoded_%s.npy'%name) # the encoder outputs
    X=np.load('./data/camb_new_processed/param_%s.npy'%name) # the input cosmological parameters
    
    
    for i in range(n_latent):
        test_name = operon_name+'_latent%d'%i
        #operon_cmb(y[:,i],X,test_name,cal_individual=False)
        operon_cmb(y[:,i],X,test_name,cal_individual=True) 

    

if __name__ == '__main__':
    main()