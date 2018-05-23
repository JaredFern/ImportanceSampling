import sample #, utils
import argparse

if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('command')

    int_args = ['alpha', 'seq_len', 'batch_cnt', 'iwal_opt']
    for arg in int_args: opt.add_argument(f"--{arg}", type=int)
    opt.add_argument('--batch_size', type=float)

    str_args = ['exp_name','method','source_file','model_file']
    for arg in str_args: opt.add_argument(f'--{arg}')

    parser = opt.parse_args()
    if parser.command == 'sample':
        if parser.method == 'ngram':
            sample.ngram_sample(train_file=parser.source_file, exp_name=parser.exp_name,
                                iwal_opt=parser.iwal_opt, alpha=parser.alpha,
                                seq_len=parser.seq_len, batch_size=parser.batch_size,
                                batch_cnt=parser.batch_cnt)
        if parser.method == 'random':
            sample.rand_sample(train_file=parser.source_file, exp_name=parser.exp_name,
                                seq_len=parser.seq_len, batch_size=parser.batch_size,
                                batch_cnt=parser.batch_cnt)
    if parser.command == 'unk_replace':
        utils.unk_replace(parser)
