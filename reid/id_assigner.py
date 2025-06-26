from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def match_features(existing_feats, new_feats):
    if not existing_feats: return list(range(len(new_feats)))

    cost_matrix = 1 - cosine_similarity(new_feats, existing_feats)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))
