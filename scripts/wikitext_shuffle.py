import random

def wikitext_shuffle(file_name, out_file):
    sequences = []
    with open(file_name,'r') as ordered:
        for line in ordered:
            tokenized = line.strip().split()
            if len(tokenized) > 0 and (tokenized[0] != "=" and tokenized[-1] != "="):
                curr_seq, parens = [], []
                for word in tokenized:
                    curr_seq.append(word)
                    if word in ('(', '[', '{'):  parens.append(word)
                    if len(parens) > 0:
                        if word == ")" and parens[-1] == "(": parens.pop()
                        elif word == "]" and parens[-1] == "[": parens.pop()
                        elif word == "}" and parens[-1] == "{": parens.pop()

                    if word == "." and len(parens) == 0:
                        sequences.append(" ".join(curr_seq))
                        curr_seq = []
                if curr_seq: sequences.append(" ".join(curr_seq))

    with open(out_file,'w') as out:
        random.shuffle(sequences)
        for line in sequences: out.write(line+"\n")

wikitext_shuffle('wiki.train.tokens', 'train.txt')
wikitext_shuffle('wiki.test.tokens', 'test.txt')
wikitext_shuffle('wiki.valid.tokens', 'valid.txt')
