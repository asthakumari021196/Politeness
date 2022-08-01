import logging
import numpy as np
import pandas as pd
import json

import tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class KeyWordFinders():
    def __init__(self,data,tag0,tag1) -> None:
        self.data = data
        self.tag0 = tag0
        self.tag1 = tag1

        
    def generate_tags(self,df_txt_class_1,
                    df_txt_class_2,
                    tag_class_1,
                    tag_class_2,
                    thresh,
                    ngram_range,
                    ):          
        stats_class_1 = TFIDFStatsGenerator(
            df_txt_class_1, tag_class_1, ngram_range=ngram_range)
        stats_class_2 = TFIDFStatsGenerator(
            df_txt_class_2, tag_class_2, ngram_range=ngram_range)

        class_1_tags = TagsGenerator(main_class_stats=stats_class_1,
                                            relative_class_stats=stats_class_2,
                                            thresh=thresh).tags

        class_2_tags = TagsGenerator(main_class_stats=stats_class_2,
                                            relative_class_stats=stats_class_1,
                                            thresh=thresh).tags

        return class_1_tags, class_2_tags                                 

    def keyword_generator(self):
        data_style_0 = self.data[self.data['p_tag'] == self.tag0]
        data_style_1 = self.data[self.data['p_tag'] == self.tag1]

        logging.info("Getting TF-IDF stats for the corpora")
        thresh=0.90
        ngram_range=(1, 2)
        
        tags_style_0, tags_style_1 = self.generate_tags(df_txt_class_1=data_style_0["txt"],
                                                   df_txt_class_2=data_style_1["txt"],
                                                   tag_class_1=self.tag0,
                                                   tag_class_2=self.tag1,
                                                   thresh=thresh,
                                                   ngram_range=ngram_range)

        # with open(f"data_keyword/{self.tag0}_tags.json", "w") as f:
        #     json.dump(tags_style_0, f)

        # with open(f"data_keyword/{self.tag1}_tags.json", "w") as f:
        #     json.dump(tags_style_1, f)
        print('tagger')
        t = tag.Tagger(data_style_1,tags_style_1, self.tag1).generate()                                  
                                          


class TFIDFStatsGenerator:

    def __init__(self, data, data_id, ngram_range):
        super().__init__()
        self.ngram_range = ngram_range
        self.data_id = data_id
        self.data = data
        self.generate()

    def get_word_counts(self):
        """Generates the counts for various n-grams for the given corpus
        
        Returns:
            a dictionary from phrase to word count
        """
        cv = CountVectorizer(ngram_range=self.ngram_range)
        cv_fit = cv.fit_transform(self.data.astype(str))
        feature_names = cv.get_feature_names() 
        X = np.asarray(cv_fit.sum(axis=0)) # sum counts across sentences
        word_to_id = {feature_names[i]: i for i in range(len(cv.get_feature_names()))}
        word_count = {}
        for w in word_to_id:
            word_count[w] = X[0, word_to_id[w]]
        return word_count

    def generate(self):
        """Generates various TFIDF related stats
        for the given data and wraps them in a namedtuple
        
        Returns:
            [type] -- [description]
        """
        logging.info("Running TfidfVectorizer")
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)

        X = vectorizer.fit_transform((self.data.astype(str)))
        feature_names = vectorizer.get_feature_names()
        id_to_word = {i: feature_names[i] for i in range(len(vectorizer.get_feature_names()))}
        word_to_id = {v: k for k, v in id_to_word.items()}
        X = np.asarray(X.mean(axis=0)).squeeze(0) # / num_docs
       
        idf = vectorizer.idf_
        counts = self.get_word_counts()
        word_to_idf = dict(zip(feature_names, idf))

        self.id_to_word = id_to_word
        self.word_to_id = word_to_id
        self.tfidf_avg = X
        self.word_to_idf = word_to_idf
        self.counts = counts


class TagsGenerator:

    def __init__(self, main_class_stats, relative_class_stats,
                 min_freq: int = 2, thresh: float = 0.90,
                 ignore_from_tags = None):
        """Generates tags for the main class relative to 
        the relative class. This is done on the basis of relative TF-IDF ratios of the words.

        Arguments:
            main_class_stats {[type]} -- [description]
            ref_class_stats {[type]} -- [description]
        
        Keyword Arguments:
            min_freq {int} -- [Minimum freq in the main class for the phrase to be considered] (default: {1})
            thresh {float} -- [The relative tf-idf scores are converted to percentiles. These percentiles are then
                               used to select the tag phrases. In this case, the cutoff for such phrases is 0.90] (default: {0.90})
            ignore_from_tags {[set]} -- [Set of words like the NER words, which might have to be ignored] (default: {None})
        """
        super().__init__()
        self.main_class_stats = main_class_stats
        self.relative_class_stats = relative_class_stats
        self.min_freq = min_freq
        self.c1_tag = main_class_stats.data_id
        self.c2_tag = relative_class_stats.data_id
        self.thresh = thresh
        self.ignore_from_tags = ignore_from_tags

        self.generate_tfidf_report()
        self.generate_relative_tags()
        

    def generate_tfidf_report(self):
        """Given TFIDF statistics on two datasets, returns a common tf-idf report. 
        The report measures various statistics on the words that appear in class_2
        
        Arguments:
            class1_tfidf_report {[TFIDFStats]} -- [TFIDFStats for class1]
            class2_tfidf_report {[TFIDFStats]} -- [TFIDFStats for class2]
        """
        report = []
        for word in self.main_class_stats.word_to_id.keys():
            if self.main_class_stats.counts[word] >= self.min_freq and word in self.relative_class_stats.word_to_id:
                    res = {}
                    res["word"] = word
                    res["freq"] = self.main_class_stats.counts[word]
                    res[f"{self.c1_tag}_mean_tfidf"] = self.main_class_stats.tfidf_avg[self.main_class_stats.word_to_id[word]]
                    res[f"{self.c2_tag}_mean_tfidf"] = self.relative_class_stats.tfidf_avg[self.relative_class_stats.word_to_id[word]]
                    res[f"{self.c1_tag}_idf"] = self.main_class_stats.word_to_idf[word]
                    res[f"{self.c2_tag}_idf"] = self.relative_class_stats.word_to_idf[word]
                    report.append(res) 
        self.report = pd.DataFrame(report)

    def generate_relative_tags(self):
        """Returns a dictionary of phrases that are important in class1 relative to
        class2
        """
        c1_over_c2 = f"{self.c1_tag}_over_{self.c2_tag}"
        c2_over_c1 = f"{self.c2_tag}_over_{self.c1_tag}"
        # tfidf_report["np_over_p"] = (tfidf_report["np_mean_tfidf"] / len(data_p_0)) / (tfidf_report["p_mean_tfidf"] /  len(data_p_9))
        self.report[c1_over_c2] = self.report[f"{self.c1_tag}_mean_tfidf"] / self.report[f"{self.c2_tag}_mean_tfidf"] #ratio of tf-idf in the two corpora

        self.report[c2_over_c1] = 1 / self.report[c1_over_c2]

        self.report[f"{self.c1_tag}_tag"] = (self.report[c1_over_c2] / self.report[c1_over_c2].sum()) ** 0.75
        # ^ add support for the small values

        self.report[f"{self.c1_tag}_tag"] = self.report[f"{self.c1_tag}_tag"] / self.report[f"{self.c1_tag}_tag"].sum()
        # ^ make a probability
        
        self.report.sort_values(by=f"{self.c1_tag}_tag", ascending=False, inplace=True)
        self.report['rank'] = self.report[f"{self.c1_tag}_tag"].rank(pct=True)
        # ^ assign percentile


        important_phrases = self.report[self.report["rank"] >= self.thresh]
        # ^ only take phrases that clear the threshold (default: 0.9)

        important_phrases["score"] = (important_phrases["rank"] - self.thresh) / (1 - self.thresh) 
        # ^ make a distribution again
        
        tags= {}
        for i, r in important_phrases.iterrows():
            tags[r["word"]] = r["score"]
        
        self.tags = tags

        if self.ignore_from_tags is not None:
            logging.info("Ignoring tags")
            self.tags = self.filter_tags_with_ignored_entities()

    def filter_tags_with_ignored_entities(self):
        res = {}
        for k, v in self.tags.items():
            if not any(k_part in self.ignore_from_tags for k_part in k.split()):
                res[k] = v
        return res        

df = pd.read_csv('cleaned_file.csv')
kg = KeyWordFinders(df,'P_0','P_9').keyword_generator()     