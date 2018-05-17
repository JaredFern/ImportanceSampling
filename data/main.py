import sample, utils
import argparse, logging

if __name__ == '__main__:
    opt = argparse.ArgumentParser()
    opt.add_argument('command')
    opt.add_argument("--entropy", action="store_true")
    opt.add_argument("--save_file", action="store_true")
    opt.add_argument("--sorted", action="store_true")
    opt.add_argument("--max_len", type=int)

    args = ['exp_name','method','source_file','seq_len', 'max_tkn'
            'model_file','sampled_file','ppl_file', 'alpha', ]
    for a in args: opt.add_argument(f'--{a}')

    opt.parse_args()
    if opt.command == 'sample':
        if opt.method == 'adaptive':
            sample.adaptive_sample(opt)
        if opt.method == 'ngram':
            sample.ngram_sample(opt)
        if opt.method == 'random':
            sample.rand_sample(opt)
    if opt.command == 'unk_replace':
        utils.unk_replace(opt)
