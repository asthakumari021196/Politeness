from cmath import nan
import numpy as np
import pandas as pd
import tqdm

class Tagger():
    def __init__(self,data,tag,tag_token_style) -> None:
        self.data = data
        self.tag_token_style = tag_token_style
        self.tag = tag

    
    def generate(self):
        print('self and dump')
        self.tag_and_dump()

    def tag_and_dump(self):
        """Iterate over the given data, tags the sentences and write out the data
        """
        print('tag and dump')

        original_sentence, tagged_sentence = [], []
        data_in = self.data
        print(data_in)
        for _, r in data_in.iterrows():
            orig = r["txt"]
            original_sentence.append(orig)
            tagged_sentence.append(Tagger.tag_sentence(orig, self.tag, self.tag_token_style).strip().replace("\n", ""))
            #polite_out.write(f"{orig}\n")
            #polite_taged_out.write(f"{taged_sent}\n")
        with open(f"data_keyword/en_parallel.en.{self.tag_token_style}", "w") as orig_out,\
             open(f"data_keyword/en_parallel.{self.tag_token_style}", "w") as taged_out:
            for orig, taged in tqdm.tqdm(zip(original_sentence, tagged_sentence), total=len(tagged_sentence)):
                if self.tag_token in taged:
                    ### ONLY WRITE OUT THE tagED DATA
                    orig_out.write(f"{orig.strip()}\n")
                    taged_out.write(f"{taged.strip()}\n")         

    @staticmethod
    def tag_sentence(sent, tag_dict, tag_token,
                      pos_weight: int = 3,
                      max_pos_indicator: int = 20,
                      concat = True):
        """Given a sentence and a dictionary from 
        tag_value to tag_probability, replaces all the words mw that are in the tag_dict
        with a probability tag_dict[mw]
        
        Arguments:
            sent {[str]}       -- [the given sentence]
            tag_dict {[dict]} -- [the tag dictionary]
            tag_token {[str]} -- [the taging token]
            dont_concat        -- [do not concat]
        
        Returns:
            [str] -- [the taged sentence]
        """
        print(len(sent))

        i = 0
        sent = sent.split()
        taged_sent = []
        prev_tag = False
            

        while len(sent) != nan and i < len(sent):
            loc = min(i // pos_weight, max_pos_indicator)
            key_bi_gram = " ".join(sent[i: i + 2])
            key_tri_gram = " ".join(sent[i: i + 3])
            key_quad_gram = " ".join(sent[i: i + 4])
            
            if key_quad_gram in tag_dict and np.random.rand() < tag_dict[key_quad_gram]:
                if not concat or not prev_tag:
                    taged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 4

            elif key_tri_gram in tag_dict and np.random.rand() < tag_dict[key_tri_gram]:
                if not concat or not prev_tag:
                    taged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 3
            elif key_bi_gram in tag_dict and np.random.rand() < tag_dict[key_bi_gram]:
                if not concat or not prev_tag:
                    taged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 2
            elif sent[i] in tag_dict and np.random.rand()< tag_dict[sent[i]]:
                if not concat or not prev_tag:
                    taged_sent.append(f"[{tag_token}{loc}]")
                prev_tag = True
                i += 1
            else:
                taged_sent.append(sent[i])
                prev_tag = False
                i += 1
        return " ".join(taged_sent)


