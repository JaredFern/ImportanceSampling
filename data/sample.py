import os, logging, kenlm, pickle
import numpy as np
import numpy.random as npr
from collections import Counter

# [sampled{seq_ind, [weight]}]

def adaptive_sample(opt):
        train_file, model_file, sampled_file=None, ppl_file = None,
                    entropy=False, exp_name='sample', alpha=1, train_cnt=1E6, seq_len=100, save=False):
        seqs, ppl, prob, sampled_ppl, total_weight, tkn_cnt = [], [], [], [], 0, 0

        ngram_model = kenlm.Model(model_file)
        if opt.sampled_file:
            sampled = pickle.load(open(opt.sampled_file,'rb'))

        if opt.entropy:
            ppl = [np.log2(ngram_model.perplexity(line)) for line in open(train_file,'r')]
        else:
            ppl = [ngram_model.perplexity(line) for line in open(train_file,'r')]

        ppl_99 = np.percentile(ppl, 99)
        mean_ppl = np.mean([p for p in ppl if p < ppl_99])
        std_ppl = np.std([p for p in ppl if p < ppl_99])

        prob = [p/std_ppl + 1 if p < ppl_99 else 1 for p in ppl]
        prob = [p if ind not in sampled else 0 for ind, p in enumerate(prob)]
        total_pr = sum(prob)
        prob[:] = [p/total_pr for p in prob]

        logger.info('Sampling training sequences.')
        sampled_ind = npr.choice(len(prob), int(2E5), p=prob, replace=False)

        for ind in sampled_ind:
            if tkn_cnt > train_cnt: break
            sampled.add(ind)
            tkn_cnt += len(seqs[ind].split())
            total_weight += len(seqs[ind].split()) * prob[ind] ** -1
            if ppl[ind] < ppl_99: sampled_ppl.append(ppl[ind])

        pickle.dump(sampled, open('sampled.bin','rb'))
        norm_const = train_cnt/total_weight

        if opt.save_sample:
            with open(f'{exp_name}.txt', 'w') as adaptive_ngram:
                for ind, line in enumerate(open(train_file,'r')):
                    if seq_ind in sampled:
                    seq = seqs[seq_ind].split()
                    while len(seq) > 0:
                        ngram.write(str(prob[seq_ind]**-1 * norm_const)
                                    + " " + " ".join( seq[:seq_len])+'\n')
                        if len(seq) > seq_len: seq = seq[seq_len:]
                        else: seq = []

def ngram_sample(train_file=['train.txt'], model_file=['iwal.arpa'], ppl_file = None,
                 iwal_opt=0, exp_name='sample', alpha=1, train_cnt=1E6, seq_len=100):
    # iwal_opt: {0: alpha, 1: z-squared, 2: z-full}
    logger.info("Model: %s (%s Tokens)", exp_name, train_cnt)
    seqs, ppl, prob, sampled_ppl = [], [], [], []
    tkn_cnt, total_weight, sampled = 0, 0, set()

    if ppl_file:
        logger.info('Loading sequence perplexity from file: %s', ppl_file)
        ppl = pickle.load(open(ppl_file,'rb'))
        for txt in train_file:
            for line in open(txt,'r'): seqs.append(line)
    elif len(model_file) == 1:
        logger.info('Caculating sequence perplexities.')
        ngram_model = kenlm.Model(model_file[0])
        for split in range(len(train_file)):
            for line in open(train_file[split], 'r'):
                seqs.append(line)
                ppl.append(ngram_model.perplexity(line.strip()))
        pickle.dump(ppl, open('ppl.bin', 'wb'))
    else:
        logger.info('Caculating sequence perplexities.')
        models = [kenlm.Model(i) for i in model_file]
        for split in range(len(train_file)):
            for line in open(train_file[split], 'r'):
                seqs.append(line)
                ppl.append(np.mean([models[i].perplexity(line).strip()
                                    for i in range(len(models)) if i != split]))
        pickle.dump(ppl, open('ppl.bin', 'wb'))

    mean_ppl, std_ppl, ppl_99 = np.mean(ppl), np.std(ppl), np.percentile(ppl, 99)
    if iwal_opt == 0:
        prob = [(p-mean_ppl)/std_ppl * alpha + 1
                if p >= mean_ppl and p < ppl_99 else 1 for p in ppl]
    elif iwal_opt == 1:
        prob = [(alpha * np.sign(p-mean_ppl)*(p-mean_ppl)**2/std_ppl**2 )+ 1
                if p >= mean_ppl and p < ppl_99 else 1 for p in ppl]
    elif iwal_opt == 2:
        prob = [alpha*(p-mean_ppl)/std_ppl + 1
                if p < ppl_99 and (p-mean_ppl)/std_ppl > -1 else 1 for p in ppl]

    total_pr = sum(prob)
    prob[:] = [p/total_pr for p in prob]

    logger.info('Sampling training sequences.')
    sampled_ind = npr.choice(len(prob), int(2E5), p=prob, replace=False)

    for ind in sampled_ind:
        if tkn_cnt > train_cnt: break
        sampled.add(ind)
        tkn_cnt += len(seqs[ind].split())
        total_weight += len(seqs[ind].split()) * prob[ind] ** -1
        if ppl[ind] < ppl_99: sampled_ppl.append(ppl[ind])

    norm_const = train_cnt/total_weight
    logger.debug('Average Sampled Sequence Perplexity: %s; Sampled StdDev: %s', np.mean(sampled_ppl), np.std(sampled_ppl))
    mean_ppl, std_ppl = np.mean([p for p in ppl if p < ppl_99]), np.std([p for p in ppl if p < ppl_99])
    logger.debug('Average Source Sequence Perplexity: %s; Source StdDev: %s\n',mean_ppl, std_ppl)

    with open(exp_name+'.txt', 'w') as ngram:
        for seq_ind in sampled:
            seq = seqs[seq_ind].split()
            while len(seq) > 0:
                ngram.write(str(prob[seq_ind]**-1 * norm_const) + " " + " ".join( seq[:seq_len])+'\n')
                if len(seq) > seq_len: seq = seq[seq_len:]
                else: seq = []

def rand_sample(train_file='train.txt', out_file='rand.txt', train_cnt=1E6, seq_len=100):
    seqmap, sampled, sampled_cnt = [], set(), 0
    with open(train_file, 'r') as f:
        seqmap = [line for line in f]
    with open(out_file, 'w') as out:
        while sampled_cnt < train_cnt:
            sampled_ind = npr.randint(0,len(seqmap))
            if sampled_ind not in sampled:
                sampled_cnt += len(seqmap[sampled_ind].split())
                sampled.add(sampled_ind)
                seq = seqmap[sampled_ind].split()
                while len(seq) > 0:
                    out.write("1.00 " + " ".join( seq[:seq_len])+"\n")
                    if len(seq) > seq_len: seq = seq[seq_len:]
                    else: seq = []
    pickle.dump(sampled, open('sampled.bin', 'wb'))
