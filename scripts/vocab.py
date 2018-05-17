from sys import argv
from collections import Counter
import argparse, logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    for root, dirs, files in os.path.walk(os.getcwd()):
        for d in dirs:
            train, test, valid = [os.path.join(d,f) for f in ['train.txt', 'test.txt', 'valid.txt']
            train_vocab, test_vocab, valid_vocab = Counter(), Counter(), Counter()

            logger.info('Experiment: %s', d)
            with open(train,'r') as train_file:
                max_len, max_seq = 0, ""
                for line in train_file:
                    train_vocab.update(line.split())
                    if len(line.split()) > max_len:
                        max_len = len(line.split())
                        max_seq = line
                logger.info('Unique Training Tokens: %s', len(train_vocab))
                logger.info('Longest sequence (%s tokens) - "%s"', max_len, max_seq)

            with open(test,'r') as test_file:
                for line in test_file: test_vocab.update(line.split())
                logger.info('Unique Test Tokens: %s', len(test_vocab))

            with open(valid,'r') as valid_file:
                for line in valid_file: valid_vocab.update(line.split())
                logger.info('Unique Valid Tokens: %s', len(valid_vocab))

            test_oov = len(set(test_vocab.keys()) - set(train_vocab.keys()))
            logger.info('OOV Test Tokens: %s (%s of Total)', test_oov,  test_oov/len(test_vocab))

            valid_oov = len(set(valid_vocab.keys()) - set(train_vocab.keys()))
            logger.info('OOV Test Tokens: %s (%s of Total)', valid_oov, valid_oov/len(valid_vocab))
