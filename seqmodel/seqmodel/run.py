
import json
import os
import time
from functools import partial

import numpy as np
import tensorflow as tf

from seqmodel import util
from seqmodel import generator as bgt
from seqmodel import dstruct as ds


__all__ = ['_no_run', 'default_training_opt', 'update_learning_rate',
           'is_done_training_early', 'run_epoch', 'train', 'decode_epoch',
           'default_decoding_opt', 'run_sampling_epoch', 'policy_gradient_opt',
           'run_collecting_epoch', 'uncond_lm_decode', 'describe_variables',
           'get_tfsession_config']


def _no_run(*args, **kwargs):
    pass


def default_training_opt():
    return {'train:max_epoch': 10, 'train:init_lr': 0.001, 'train:clip_gradients': 10.0,
            'train:optim_class': 'tensorflow.train.AdamOptimizer', 'lr:min_lr': 1e-6,
            'lr:start_decay_at': 1, 'lr:decay_every': 1, 'lr:decay_factor': 1.0,
            'lr:imp_ratio_threshold': 0.0, 'lr:imp_wait': 2}


def policy_gradient_opt():
    return {'pg:enable': False, 'pg:discount': 0.9, 'pg:sample_logprob': False}


def default_decoding_opt():
    return {'decode:greedy': False, 'decode:outpath': 'decode_out.txt',
            'decode:num_samples': 1}


def update_learning_rate(
        set_lr_fn, train_state, min_lr=1e-6, start_decay_at=1, decay_every=1,
        decay_factor=1.0, imp_ratio_threshold=0, imp_wait=2):
    old_lr = train_state.learning_rate
    new_lr = old_lr
    if train_state.cur_epoch < start_decay_at or train_state.cur_epoch == 0:
        # waiting to start decay
        set_lr_fn(new_lr)
        return old_lr  # EARLY RETURN!
    if decay_every > 0 and train_state.cur_epoch % decay_every == 0:
        # schedule decay
        new_lr = old_lr * decay_factor
    elif imp_ratio_threshold > 0:
        # adaptive decay
        imp_ratio = train_state.cur_eval / (train_state.last_imp_eval + 1e-12)
        if imp_ratio < imp_ratio_threshold:
            train_state.last_imp_eval = train_state.cur_eval
            train_state.last_imp_epoch = train_state.cur_epoch
            train_state.imp_wait = 0
        else:
            train_state.imp_wait += 1
            if train_state.imp_wait >= imp_wait:
                new_lr = old_lr * decay_factor
                if decay_factor < 1.0 and new_lr > min_lr:
                    train_state.imp_wait = 0
    new_lr = max(new_lr, min_lr)
    set_lr_fn(new_lr)
    train_state.learning_rate = new_lr
    return new_lr


def is_done_training_early(train_state, imp_wait=2, min_lr=1e-04):
    return train_state.imp_wait >= imp_wait and train_state.learning_rate <= min_lr


def run_epoch(
        sess, model, batch_iter, train_op=None, train_state=None, begin_step_fn=None,
        end_step_fn=None, _extra_data=None):
    info = ds.RunningInfo()
    if train_op:
        run_fn = partial(model.train, sess, train_op=train_op)
    else:
        run_fn = partial(model.evaluate, sess)
    state = None
    for batch in batch_iter():
        if begin_step_fn is not None:
            begin_step_fn(step_info=info, train_state=train_state)
        if _extra_data is not None:
            batch = _extra_data.pop(0)
        result, __ = run_fn(batch.features, batch.labels, state=state,
                            fetch_state=batch.keep_state)
        if batch.keep_state:
            result, state = result  # ds.OutputStateTuple
        else:
            state = None
        if end_step_fn is not None:
            end_step_fn(step_info=info, train_state=train_state)
        info.update_step(result, batch.num_tokens)
    info.end()
    return info


def run_collecting_epoch(
        sess, model, batch_iter, collect_keys, collect_fn, train_op=None,
        train_state=None):
    info = ds.RunningInfo()
    if train_op:
        run_fn = partial(model.train, sess, train_op=train_op, extra_fetch=collect_keys)
    else:
        run_fn = partial(model.evaluate, sess, extra_fetch=collect_keys)
    state = None
    for batch in batch_iter():
        result, collect = run_fn(
            batch.features, batch.labels, state=state, fetch_state=batch.keep_state)
        collect_fn(batch, collect)
        if batch.keep_state:
            result, state = result  # ds.OutputStateTuple
        else:
            state = None
        info.update_step(result, batch.num_tokens)
    info.end()
    return info


def _acc_discounted_rewards(rewards, discount_factor, baseline=1e-4):
    R = np.zeros_like(rewards)
    r_tplus1 = np.zeros([rewards.shape[1]])
    for i in range(len(rewards) - 1, -1, -1):
        R[i, :] = rewards[i, :] + discount_factor * r_tplus1
        r_tplus1 = R[i, :]
    return R - baseline


def run_sampling_epoch(
        sess, model, batch_iter, train_op=None, reward_fn=None, greedy=False,
        discount_factor=0.9, pack_data_fn=None, return_fn=_acc_discounted_rewards,
        with_score=False, return_feed_fn=None, train_state=None):
    if pack_data_fn is None:
        def pack_data_fn(batch, sample, ret):
            # assume seq2seq data
            pg_batch = bgt.get_batch_data(
                batch, sample, input_key='dec_inputs', seq_len_key='dec_seq_len')
            return pg_batch, ret
    assert reward_fn is not None, 'reward_fn must not be None.'
    decode_fn = model.decode_sampling
    if greedy and with_score:
        decode_fn = model.decode_greedy_w_score
    elif greedy:
        decode_fn = model.decode_greedy
    elif with_score:
        decode_fn = model.decode_sampling_w_score
    train_result, score = None, None
    info = ds.RunSamplingInfo()
    for batch in batch_iter():
        sample, __ = decode_fn(sess, batch.features)
        if with_score:
            sample, score = sample
        reward, avg_reward = reward_fn(sample, batch, sample_score=score)
        num_tokens = batch.num_tokens
        if train_op is not None:
            ret = return_fn(reward, discount_factor)
            train_batch, ret = pack_data_fn(batch, sample, ret)
            return_feed_fn(ret)
            train_result, __ = model.train(
                sess, train_batch.features, train_batch.labels, train_op=train_op)
            num_tokens = train_batch.num_tokens
        info.update_step(avg_reward, num_tokens, train_result)
    info.end()
    return info


def train(
        train_run_epoch_fn, logger, max_epoch=1, train_state=None, init_lr=1.0,
        valid_run_epoch_fn=None, begin_epoch_fn=None, end_epoch_fn=None):
    if train_state is None:
        train_state = ds.TrainingState()
        train_state.learning_rate = init_lr
    stop_early = False
    for epoch in range(train_state.cur_epoch, max_epoch):
        if begin_epoch_fn is not None:
            begin_epoch_fn(train_state)
        logger.info(train_state.summary(mode='train'))
        state_info = train_run_epoch_fn(train_state=train_state)
        logger.info(state_info.summary(mode='train'))
        if valid_run_epoch_fn is not None:
            valid_info = valid_run_epoch_fn()
            logger.info(valid_info.summary(mode='valid'))
            state_info = valid_info
        train_state.update_epoch(state_info)
        if end_epoch_fn is not None:
            stop_early = end_epoch_fn(train_state)
        if stop_early:
            break
    else:
        logger.info(f'Maximum epoch reach at {train_state.cur_epoch}')
    return train_state


def decode_epoch(sess, model, batch_iter, greedy=False, num_samples=1):
    decode_fn = model.decode_sampling
    if greedy:
        decode_fn = model.decode_greedy
    for batch in batch_iter():
        samples = []
        for __ in range(num_samples):
            sample, __ = decode_fn(sess, batch.features)
            samples.append(sample)
        yield batch, samples


def uncond_lm_decode(sess, model, feature_seed, greedy=False, vocabs=None):
    state = None
    feature = feature_seed
    dec_mode = 'dec_max_id' if greedy else 'dec_sample_id'
    while True:
        result, __ = model.predict(
            sess, feature, predict_key=dec_mode, fetch_state=True, state=state)
        output, state = result
        feature = feature._replace(inputs=output[[-1], :])
        feature.seq_len[:] = 1
        yield output, vocabs


def describe_variables(variables):
    var_desc = []
    total_params = 0
    for v in variables:
        var_desc.append(f'- {v.name}, {v.shape}')
        total_params += int(np.prod(v.shape))
    var_desc.append(f'Total parameters: {total_params:5E}')
    return '\n'.join(var_desc)

def get_tfsession_config(is_gpu, num_threads=12):
    if is_gpu:
        return tf.ConfigProto(
            intra_op_parallelism_threads=num_threads,
            inter_op_parallelism_threads=num_threads)
    else:
        return tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=num_threads,
            inter_op_parallelism_threads=num_threads)
