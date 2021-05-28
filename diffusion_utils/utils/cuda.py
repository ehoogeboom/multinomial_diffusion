import os


def set_cuda_vd(gpu_ids, verbose=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(id) for id in gpu_ids)
    if verbose: print("CUDA_VISIBLE_DEVICES = {}",format(os.environ["CUDA_VISIBLE_DEVICES"]))
