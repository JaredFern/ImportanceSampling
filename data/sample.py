import os, logging, kenlm, pickle, subprocess
import numpy as np
import numpy.random as npr
from collections import Counter

def ngram_sample(train_file, exp_name, iwal_opt=0, alpha=1, seq_len=40, batch_size=5E5, batch_cnt=1):
    print ("Training ngram on random starter set")
    sentences = [sent for sent in open(train_file, 'r')]

    rand_sample(train_file=train_file, exp_name=f'{exp_name}/train-0.txt', batch_size=batch_size, seq_len=seq_len)

    count_cmd = f"ngram-count -text {exp_name}/train-0.txt -write {exp_name}/counts.arpa -lm {exp_name}/combined.arpa -order 5 -unk -kndiscount5;"
    count_subproc = subprocess.Popen(count_cmd, stdout=subprocess.PIPE, shell=True)
    count_subproc.wait()
    model_file = f'{exp_name}/combined.arpa'

    for batch in range(1,batch_cnt):
        print (f"\nBatch: {batch}")
        prob, sampled_ppl = [], []
        tkn_cnt, total_weight, sampled = 0, 0, set()

        ngram_model = kenlm.Model(model_file)
        print (f"Calculating sequence perplexities with: {model_file}")
        sentence_ppl = [ngram_model.perplexity(sent) for sent in sentences]

        print (f"Sampling Sequences.")
        mean_ppl, std_ppl, ppl_99 = np.mean(sentence_ppl), np.std(sentence_ppl), np.percentile(sentence_ppl, 99)
        if iwal_opt == 0: # Z-alpha
            prob = [(p-mean_ppl)/std_ppl * alpha + 1
                    if p >= mean_ppl and p < ppl_99 else 1 for p in sentence_ppl]
        elif iwal_opt == 1: # Z-squared
            prob = [(alpha * np.sign(p-mean_ppl)*(p-mean_ppl)**2/std_ppl**2 )+ 1
                    if p >= mean_ppl and p < ppl_99 else 1 for p in sentence_ppl]
        elif iwal_opt == 2: # Z-Full
            prob = [alpha*(p-mean_ppl)/std_ppl + 1
                    if p < ppl_99 and (p-mean_ppl)/std_ppl > -1 else 1 for p in sentence_ppl]

        total_pr = sum(prob)
        prob[:] = [p/total_pr for p in prob]

        sampled_ind = npr.choice(len(prob), int(2E5), p=prob, replace=False)
        for ind in sampled_ind:
            if tkn_cnt > batch_size: break
            sampled.add(ind)
            tkn_cnt += len(sentences[ind].split())
            total_weight += len(sentences[ind].split()) * prob[ind] ** -1
            if sentence_ppl[ind] < ppl_99: sampled_ppl.append(sentence_ppl[ind])

        norm_const = batch_size/total_weight
        with open(f'{exp_name}/train-{batch}.txt', 'w') as ngram:
            for seq_ind in sampled:
                sent = sentences[seq_ind].split()
                while len(sent) > 0:
                    ngram.write(str(prob[seq_ind]**-1 * norm_const) + " " + " ".join( sent[:seq_len])+'\n')
                    if len(sent) > seq_len: sent = sent[seq_len:]
                    else: sent = []

        print (f"Training ngram on batch: {batch}")
        count_cmd = f"ngram-count -text {exp_name}/train-{batch}.txt -read {exp_name}/counts.arpa -write {exp_name}/counts.arpa -lm {model_file} -order 5 -kndiscount5 -unk;"
        count_subproc = subprocess.Popen(count_cmd, stdout=subprocess.PIPE, shell=True)
        count_subproc.wait()

def rand_sample(train_file, exp_name, seq_len=40, batch_size=1E6, batch_cnt = 1):
    sampled_cnt = 0
    os.makedirs(exp_name, exist_ok=True)
    seqmap = [line for line in open(train_file,'r')]
    for batch in range(batch_cnt):
        with open(f'{exp_name}/train-{batch}.txt', 'w') as out:
            while sampled_cnt < batch_size:
                sampled_ind = npr.randint(0,len(seqmap))
                sampled_cnt += len(seqmap[sampled_ind].split())
                seq = seqmap[sampled_ind].split()
                while len(seq) > 0:
                    out.write("1.0000 " + " ".join( seq[:seq_len])+"\n")
                    if len(seq) > seq_len: seq = seq[seq_len:]
                    else: seq = []
