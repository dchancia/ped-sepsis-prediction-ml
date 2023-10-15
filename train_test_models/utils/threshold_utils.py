import numpy as np
from sklearn.metrics import recall_score
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def find_thresh (y, probas, i, reps):
    """
    Find new threshold to have a fixed sensitivity of 0.85.
    """

    flag = 0
    lower_limit = 0.0
    upper_limit = 1.0
    reps = 101
    start_time = datetime.now()
    wait = 2

    while (flag == 0) and ((((datetime.now() - start_time).total_seconds()) / 60) <= wait):

        thresh = [round(x, 200) for x in list(np.linspace(upper_limit, lower_limit, num=reps))]

        for t in thresh:

            upd_preds = [1 if x >= t else 0 for x in list(probas[:,1])]
            sens = recall_score(y, upd_preds)
            
            if sens >= 0.82 and sens <= 0.84:
                upper_limit = t
            if sens >= 0.86 and sens <= 0.88:
                lower_limit = t
            if sens >= 0.846 and sens <= 0.854:
                flag = 1
                break
            else:
                t = None
        
        if t == None:
            reps += 1000
        
    return t



def get_coord (x, y):
    """
    Find coordinates for 0.85 sensitivity for ROC Curve
    """
    for elem_y in list(y):
        if elem_y >= 0.85:
            break
    return list(x)[list(y).index(elem_y)], elem_y   





