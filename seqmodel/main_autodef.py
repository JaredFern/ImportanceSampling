import time
import os
from functools import partial

import numpy as np

from _main import sq
from _main import mle
from _main import policy_gradient
from _main import decode

_ENCODE = True

if __name__ == '__main__':
    start_time = time.time()
    group_default = {'model': sq.AutoSeqModel.default_opt(),
                     'train': sq.default_training_opt(),
                     'pg': sq.policy_gradient_opt(),
                     'decode': sq.default_decoding_opt()}
    parser = sq.get_common_argparser('main_autodef.py')
    sq.add_arg_group_defaults(parser, group_default)
    opt, groups = sq.parse_set_args(parser, group_default, dup_replaces=('enc:', 'dec:'))
    logger, all_opt = sq.init_exp_opts(opt, groups, group_default)
    opt, model_opt, train_opt, decode_opt, pg_opt = all_opt

    def data_fn():
        dpath = partial(os.path.join, opt['data_dir'])
        enc_vocab = sq.Vocabulary.from_vocab_file(dpath('enc_vocab.txt'))
        dec_vocab = sq.Vocabulary.from_vocab_file(dpath('dec_vocab.txt'))
        data_fn = partial(sq.read_lseq2seq_data, in_vocab=dec_vocab, out_vocab=dec_vocab,
                          l_vocab=enc_vocab)
        data = [data_fn(sq.read_lines(dpath(f), token_split=' ', part_split='\t',
                                      part_indices=(-1, -1, 0)))
                for f in (opt['train_file'], opt['valid_file'], opt['eval_file'])]

        batch_iter = partial(sq.lseq2seq_batch_iter, batch_size=opt['batch_size'],
                             shuffle=not _ENCODE)
        return data, batch_iter, (enc_vocab, dec_vocab)

    if opt['command'] == 'decode':
        with open(decode_opt['decode:outpath'], 'w') as ofp:
            def decode_batch(batch, samples, vocabs):
                b_enc = vocabs[0].i2w(batch.features.enc_inputs.T)
                b_enc_len = batch.features.enc_seq_len
                for b_samples in samples:
                    b_seq_len = sq.find_first_min_zero(b_samples)
                    for enc, enc_len, dec, dec_len in zip(
                            b_enc, b_enc_len, b_samples.T, b_seq_len):
                        if enc[0] == '</s>':
                            continue
                        enc_text = ' '.join(enc[:enc_len - 1])
                        dec_text = ' '.join(vocabs[1].i2w(dec[:dec_len]))
                        ofp.write(f'{enc_text}\t{dec_text}\n')
            decode(opt, model_opt, decode_opt, decode_batch, logger,
                   data_fn, sq.AutoSeqModel)
    else:
        if pg_opt['pg:enable']:
            policy_gradient(opt, model_opt, train_opt, pg_opt, logger, data_fn,
                            sq.AutoSeqModel)
        else:
            if _ENCODE:
                exp_path = partial(os.path.join, opt['exp_dir'])
                states = []
                batchs = []
                embs = []

                def collect(batch, data):
                    batchs.append(batch)
                    states.append(data)
                    embs.append(data[0])
                # collect_keys = ['bridge.main', 'bridge.logsigma']
                collect_keys = ['bridge.emb_output']
                eval_run_fn = partial(
                    sq.run_collecting_epoch,
                    collect_keys=collect_keys, collect_fn=collect)
                mle(opt, model_opt, train_opt, logger, data_fn, sq.AutoSeqModel,
                    eval_run_fn=eval_run_fn)

                import pickle
                with open(exp_path('batch_emb.pick'), mode='wb') as ofp:
                    pickle.dump(zip(batchs, states), ofp)
                embs = np.vstack(embs)
                np.save(exp_path('emb.npy'), embs)
            else:
                mle(opt, model_opt, train_opt, logger, data_fn, sq.AutoSeqModel)
    logger.info(f'Total time: {sq.time_span_str(time.time() - start_time)}')
