import numpy as np
def datagen_aggregated(gen, X, Y_original, Y_aggregate, batch_size):
    """Takes full image data and corresponding ground-truth
    and aggregate labels, in that order, and returns a batch of them."""
    genY_original = gen.flow(X, Y_original, seed=9, batch_size=batch_size)
    genY_aggregate = gen.flow(X, Y_aggregate, seed=9, batch_size=batch_size)
    
    while True:
        Xi, Yi1 = genY_original.next()
        Xi, Yi2 = genY_aggregate.next()
        yield Xi, [Yi1, Yi2]

# to call
#final_gen = two_label_datagen_aggregated(datagen, x_test,
#                                         y_test,
#                                         y_agg,
#                                         batch_size)
    

def datagen_sparse(gen, X, x_idx, Y_original, Y_sparse, batch_size):
    """Takes original image data and observation indices and original 
    ground-truth labels and sparse labels, iterates over batches of 
    observation indices and sparse labels, and returns batched observation
    images, batched ground-truth labels, and batched sparse labels."""
    
    genY_sparse = gen.flow(x_idx, Y_sparse, seed=9, batch_size=batch_size)
    
    while True:
        x_idx_i, Y_sparse_i = genY_sparse.next()
        
        yield X[x_idx_i], [Y_original[x_idx_i], Y_sparse_i]

# to call
#    final_gen = two_label_datagen(datagen, x_test,
#                                      x_idx,
#                                      y_test,
#                                      y_sparse,
#                                      batch_size)
