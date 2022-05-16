import subprocess
import argparse
import os


def gen_data_multiround(args):
    nums = [1000, 3000, 5000, 10000, 30000, 50000, 100000, 300000]
    for i, num in enumerate(nums):
        out_path = os.path.join(args.out_dir, f'train{num}.pkl')
        command = 'python datasets.py --env-path {} --out-path {} --num-data {}'.format(args.env_path, out_path, num)
        subprocess.Popen(command, shell=True).wait()
    print('Finish generating data.')


def main(args):
    nums = [1000, 3000, 5000, 10000, 30000, 50000, 100000, 300000]
    for i, num in enumerate(nums):
        train_path = os.path.join(args.out_dir, f'train{num}.pkl')
        model_name = f'{args.model_type}_model{num}.pt'
        test_path = '../data_mitigate/testset_randomcirc.pkl'
        command = 'python train_{}.py --logdir {} --train-path {} --test-path {} --save-name {}'.format(
            args.model_type, '../runs/ibmq_multiround', train_path, test_path, model_name)
        subprocess.Popen(command, shell=True).wait()
    print('Multi-training finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-path', default='../environments/ibmq_random.pkl', type=str)
    parser.add_argument('--out-dir', default='../data_multiround', type=str)
    parser.add_argument('--model-type', default='gan', type=str)
    args = parser.parse_args()

    # gen_data_multiround(args)
    main(args)