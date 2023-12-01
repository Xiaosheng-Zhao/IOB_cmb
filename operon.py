import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import InfixFormatter, FitLeastSquares, MSE
import csv
import argparse
import string
import os
import sys


def split_by_punctuation(s):
    pun = string.punctuation.replace('_', '') # allow underscores in variable names
    pun = string.punctuation.replace('.', '') # allow full stops
    pun = pun + ' '
    where_pun = [i for i in range(len(s)) if s[i] in pun]
    if len(where_pun) > 0:
        split_str = [s[:where_pun[0]]]
        for i in range(len(where_pun)-1):
            split_str += [s[where_pun[i]]]
            split_str += [s[where_pun[i]+1:where_pun[i+1]]]
        split_str += [s[where_pun[-1]]]
        if where_pun[-1] != len(s) - 1:
            split_str += [s[where_pun[-1]+1:]]
    else:
        split_str = [s]
    return split_str


def is_float(string):
    try:
        float(eval(string))
        return True
    except:
        return False
    

def replace_floats(s):
    split_str = split_by_punctuation(s)
    values = []
    for i in range(len(split_str)):
        if is_float(split_str[i]) and "." in split_str[i]:
            values.append(float(split_str[i]))
            split_str[i] = f'a{len(values)-1}'
        elif len(split_str[i]) > 1 and split_str[i][-1] == 'e' and is_float(split_str[i][:-1]):
            if split_str[i+1] in ['+', '-']:
                values.append(float(''.join(split_str[i:i+3])))
                split_str[i] = f'a{len(values)-1}'
                split_str[i+1] = ''
                split_str[i+2] = ''
            else:
                assert split_str[i+1].is_digit()
                values.append(float(''.join(split_str[i:i+2])))
                split_str[i] = f'a{len(values)-1}'
                split_str[i+1] = ''
    replaced = ''.join(split_str)
    return replaced, values


def process_tree(reg, tree, X, y):
    
    k = 0
    uniquefun = []
    for node in tree.Nodes:
        if node.IsVariable:
            uniquefun.append(str(node.HashValue))
        else:
            uniquefun.append(node.Name)
        if node.IsVariable:
            k += 3
            uniquefun.append('*')
            uniquefun.append('constant')
        else:
            k += 1
    uniquefun = list(set(uniquefun))
    
    y_pred = reg.evaluate_model(tree, X)
    scale, offset = FitLeastSquares(y_pred, y)

    if (np.isfinite(offset) and np.isfinite(scale)):
        y_pred = scale * y_pred + offset
        mse = np.mean((y - y_pred)**2)
        r2 = r2_score(y, y_pred)
    else:
        mse = np.nan
        r2 = np.nan
    
    return k, uniquefun, scale, offset, mse, r2


def operon_cmb(y,X,test_name,cal_individual):
    '''
    # y: the encoder outputs,[batch,6]
    # X: the cosmological parameters,[batch,6]
    # test_name: a specific name for this test
    # cal_individual: whether to calculate the statistics related to the individual expressions (might encounter NaN bug)
    '''
    
    # Make SR class
    reg = SymbolicRegressor(
            allowed_symbols='add,sub,mul,div,sin,constant,variable',
            offspring_generator='basic',
            optimizer_iterations=1000,
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
    reg.fit(X.copy(), y)
    
    compute_mse = MSE()
    
    # Output pareto front to file 
    print('Outputting pareto')
    with open('./data/sr/pareto_%s.csv'%test_name, 'w') as f:
        
        writer = csv.writer(f, delimiter=';')

        writer.writerow(['length',
                        'r2',
                        'mse',
                        'scale',
                        'offset',
                        'nuniqueops',
                        'complexity',
                        'nparam',
                        'fun',
                        'theta'
        ])

        pred_ind = np.zeros((len(reg.pareto_front_),len(y)))
        pareto_mse = np.zeros((len(reg.pareto_front_)))

        print(f'Outputting {len(reg.pareto_front_)} individuals on Pareto front')

        for i in range(len(reg.pareto_front_)):
            tree = reg.pareto_front_[i]['tree']
            k, uniquefun, scale, offset, mse, r2 = process_tree(reg, tree, X, y)
            replaced, values = replace_floats(reg.pareto_front_[i]['model'])
            writer.writerow([tree.Length,
                             r2,
                             mse,
                             scale,
                             offset,
                             len(uniquefun),
                             k,
                             len(values),
                             replaced,
                             str(values)
                            ])
    
            
            y_pred = reg.evaluate_model(tree, X)
            scale, offset = FitLeastSquares(y_pred, y)
            pred_ind[i] = scale * y_pred + offset
            pareto_mse[i] = mse
            #np.save('./data/pred_individuals5_%d.npy'%count,y_pred_test)
    
        np.save('./data/sr/pred_pareto_%s.npy'%test_name,pred_ind)
        np.save('./data/sr/mse_pareto_%s.npy'%test_name,pareto_mse)
        
    if cal_individual==True:
    
        # Output currently considered individuals to file
        print(f'Outputting {len(reg.individuals_)} individuals in population')
            
        with open('./data/sr/individuals_%s.csv'%test_name, 'w') as f:

            writer = csv.writer(f, delimiter=';')
            writer.writerow(['length',
                            'r2',
                            'mse',
                            'scale',
                            'offset',
                            'nuniqueops',
                            'complexity',
                            'nparam',
                            'fun',
                            'theta'
            ])

            pred_ind = np.zeros((reg.population_size,len(y)))
                
            for ind in reg.individuals_[:reg.population_size]:
                k, uniquefun, scale, offset, mse, r2 = process_tree(reg, ind.Genotype, X, y)

                # Get name but block printing to sys.stderr
                sys.stderr = open(os.devnull, 'w')
                s = reg.get_model_string(ind.Genotype, 10)
                sys.stderr = sys.__stderr__

                replaced, values = replace_floats(s)
                writer.writerow([ind.Genotype.Length,
                             r2,
                             mse,
                             scale,
                             offset,
                             len(uniquefun),
                             k,
                             len(values),
                             replaced,
                             str(values)
                            ])
                
                y_pred = reg.evaluate_model(ind.Genotype, X)
                scale, offset = FitLeastSquares(y_pred, y)
                if np.isfinite(scale) and np.isfinite(offset):
                    pred_ind[i] = scale * y_pred + offset
                else:
                    pred_ind[i,:] = np.nan
                
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
        
def test():
    
    # Make some mock data
    nx = 100
    X = np.array([np.linspace(0, 1, nx), np.linspace(0, 1, nx)]).T
    y = 3.0 * X[:,0] * np.cos(np.pi * np.sqrt(0.5 * X[:,1] ** 2 + X[:,0] ** 2))
    yerr = 0.05 * y
    y = y + np.random.normal(nx) * yerr
    operon_cmb(y,X,'test',cal_individual=True) 
    

if __name__ == '__main__':
    # main()
    test()