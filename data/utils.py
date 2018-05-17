from collections import Counter
import logger, kenlm, sample

def unk_replace(src_files=['train.txt','test.txt','valid.txt'], vocab_size=25e4):
    vocab_cnt, vocab_size = Counter(), int(vocab_size)
    logger.info("Building vocabulary and word frequencies.")
    for fname in src_files:
        for line in open(fname,'r'):
            for word in line.split():
                vocab_cnt[word] += 1

    sorted_vocab = vocab_cnt.most_common(vocab_size)
    tkn_cnt = sum(vocab_cnt.values())

    for word_freq in vocab_cnt.most_common()[vocab_size:]:
        vocab_cnt["<unk>"] += word_freq[1]
        del vocab_cnt[word_freq[0]]

    with open('vocab.txt', 'w') as vocab:
        vocab.write("</s>\n")
        for word, cnt in vocab_cnt.most_common():
            vocab.write(f"{word} {cnt}\n")

    for fname in src_files:
        with open(fname[:-4]+'_unk.txt','w') as dst:
            for line in open(fname, 'r'):
                curr_seq = [word if word in vocab_cnt else "<unk>" for word in line.split()]
                dst.write(" ".join(curr_seq) + '\n')

def ppl_stats(fname, model,weights=False):
    ngram = kenlm.Model(model)
    if weights:
        ppl = [ngram.perplexity(line.split(" ",1)[1]) for line in open(fname,'r')]
    else: ppl = [ngram.perplexity(line) for line in open(fname,'r')]
    ppl_99 = np.percentile(ppl, 99)
    ppl = [p for p in ppl if p < ppl_99]
    return (np.mean(ppl), np.std(ppl))

def sample_all(train_txt, models, train_cnt=1e6, ppl_file=None):
    '''train_txt: list of train files, models: list of ngram arpa's'''
    rand_sample(train_file=train_txt[0], train_cnt=train_cnt)
    ngram_sample(train_file=train_txt, model_file= models,ppl_file=ppl_file,
                exp_name='one-alpha', train_cnt=train_cnt, alpha=1, iwal_opt=0)
    ngram_sample(train_file= train_txt, model_file= models, ppl_file='ppl.bin',
                exp_name='two-alpha', train_cnt=train_cnt, alpha=2, iwal_opt=0)
    ngram_sample(train_file= train_txt, model_file= models, ppl_file='ppl.bin',
                exp_name='four-alpha', train_cnt=train_cnt, alpha=4, iwal_opt=0)
    ngram_sample(train_file= train_txt, model_file= models, ppl_file='ppl.bin',
                exp_name='half-alpha', train_cnt=train_cnt, alpha=0.5, iwal_opt=0)
    ngram_sample(train_file= train_txt, model_file= models, ppl_file='ppl.bin',
                exp_name='z-squared', train_cnt=train_cnt, alpha=1, iwal_opt=1)
    ngram_sample(train_file= train_txt, model_file= models, ppl_file='ppl.bin',
                exp_name='z-full', train_cnt=train_cnt, alpha=1, iwal_opt=2)
