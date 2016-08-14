import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean
import glob


def merge_several_folds_mean_create_submission(sub_list):
    predictions_0 = pd.read_csv(sub_list[0])
    test_id = predictions_0.img
    predictions_0.drop('img',inplace = True,axis=1)
    predictions = predictions_0.values
    for i in xrange(1,len(sub_list)):
        next_p = pd.read_csv(sub_list[i])
        next_p.drop('img',inplace = True,axis=1)
        predictions += next_p.values
    predictions /= len(sub_list)
    df = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', test_id)
    df.to_csv('subm/submission_22.csv', index = False)


def merge_several_folds_mean_create_submission_gmean(num_folds):
    predictions_0 = pd.read_csv('subm/0_submission_2016-06-11-23-14.csv')
    test_id = predictions_0.img
    predictions_0.drop('img',inplace = True,axis=1)
    p = []
    p.append(predictions_0.values)
    for i in xrange(num_folds - 1):
        next_p = pd.read_csv('subm/{ind}_submission_2016-06-12-14-32.csv'.format(ind=i))
        next_p.drop('img',inplace = True,axis=1)
        p.append(next_p.values)
    predictions = gmean(p, axis = 0)
    df = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df.insert(0, 'img', test_id)
    df.to_csv('subm/sub_test3_gmean.csv', index = False)


if __name__ == '__main__':
    l = []
    for i in glob.glob('subm/*'):
        try:
            int(i.split('/')[1].split('_')[0])
            l.append(i)
        except:
            pass
    for i in l:
        if i.split('-')[2] == '09':
            l.remove(i)

    merge_several_folds_mean_create_submission(l)
