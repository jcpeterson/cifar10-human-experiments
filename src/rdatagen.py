import numpy as np
import time


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
#final_gen = datagen_aggregated(datagen, x_test,
#                                         y_test,
#                                         y_agg,
#                                         batch_size)
    

def datagen_sparse(gen, X, x_idx, Y_original, Y_sparse, batch_size, num_splits=50):
    """Takes original image data and observation indices and original 
    ground-truth labels and sparse labels, iterates over batches of 
    observation indices and sparse labels, and returns batched observation
    images, batched ground-truth labels, and batched sparse labels."""
    aug_data_iters_sparse = []
    aug_data_iters_original = []
    x_idx_splits = np.array_split(x_idx, num_splits)
    Y_sparse_splits = np.array_split(Y_sparse, num_splits)
    count = 0
    for x_i, Y_i in tuple(zip(x_idx_splits, Y_sparse_splits)):
#        time.sleep(5)
#        print("split set: ", count)
#        time.sleep(5)
        gen_s = gen.flow(np.squeeze(X[x_i]), Y_i)
        gen_o = gen.flow(np.squeeze(X[x_i]), Y_original[x_i])

        while True:
            X_i, Y_sparse_i = gen_s.next()
            X_i, Y_original_i = gen_s.next()
        
            yield X_i, [Y_sparse_i, Y_original_i]

# to call
#    final_gen = datagen_sparse(datagen, x_test,
#                                      x_idx,
#                                      y_test,
#                                      y_sparse,
#                                      batch_size)
