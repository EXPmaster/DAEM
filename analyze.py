import numpy as np
import pickle


if __name__ == '__main__':
    data_path = 'data_mitigate/testset_vqe4l.pkl'
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    print(len(dataset))

    deviations = []
    for params, obs, pos, scale, exp_noisy, exp_ideal in dataset:
        if scale == 0.01:
            deviations.append(abs(exp_noisy - exp_ideal))
    mean_val = round(np.mean(deviations), 6)
    print('deviation in original dataset: {:.6f}'.format(mean_val))
    
