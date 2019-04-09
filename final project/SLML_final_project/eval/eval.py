from bleu import Bleu
from cider import Cider
import json
class COCOEvalCap:
    def __init__(self):
        self.eval = {}


    def evaluate(self):
        cap = open(r'results.txt')
        cap_ = []
        for line in cap:
            line = line.split(' ')
            line[len(line)-1] = '.'
            del line[0]
            print(line)
            cap_.append(line)
        gts = {}
        res = {}
        f = open("cap_flickr30k.json")
        captions = json.load(f)
        f1 = open("dic_flickr30k.json")
        dics = json.load(f1)
        dics = dics['images']
        pos = 0
        for i in range(0, len(dics), 1):
            if dics[i]['split'] == 'test':
                caption_1 = []
                caption_2 = []
                caption_1.append(captions[i][0]['caption'])
                res[dics[i]['id']] = caption_1
                caption_2.append(cap_[pos])
                caption_2.append(cap_[pos])
                gts[dics[i]['id']] = caption_2
                pos = pos + 1

        # =================================================
        # Set up scorers
        # =================================================

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print ('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    print ("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                print ("%s: %0.3f"%(method, score))


    def setEval(self, score, method):
        self.eval[method] = score



a = COCOEvalCap()
a.evaluate()