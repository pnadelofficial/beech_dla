# imports
import pandas as pd
import numpy as np 
from scipy.stats import pearsonr, norm
from tqdm import tqdm
import spacy
from textacy.extract import ngrams
import collections
from string import punctuation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import ipywidgets as widgets
from IPython.display import display
import textwrap
import ast
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
tqdm.pandas()

class DLAnalyzer:
    """
    Abstraction for conducting differential language analysis

    Takes in a dataframe and an optional name. This dataframe can 
    be the DASS dataframe we've been working from, but can also be
    any subset of that dataframe. 
    """
    def __init__(self, df, remove_stops=True, lemmatize=False, min_word_freq=1, min_narr_size=5, alpha=.05, name=None):
        self.remove_stops = remove_stops
        self.lemmatize = lemmatize
        self.min_word_freq = min_word_freq
        self.min_narr_size = min_narr_size
        self.alpha = alpha

        df = df.dropna(subset='Narrative')
        df = df.loc[df.Narrative.str.len() > self.min_narr_size].reset_index(drop=True)
        df = df.reset_index() # need 'index' column for `generate_group_norm_df`
        self.df = df

        self.name = name
        self.nlp = spacy.load('en_core_web_sm')
    
    def create_group_norms(self, ID, text):
        """
        Calculates group norm (total occurence of token divided by 
        the amount of tokens in a given narrative) for each token

        This function first lowers the text in each narrative section,
        then it gets all of the uni-, bi- and trigrams in the narrative.
        Then it does the count and calculation of group norm.
        """
        doc = self.nlp(text.lower())
        if self.lemmatize:
            if self.remove_stops: uni = [t.lemma_ for t in doc if not t.is_stop]
            else: uni = [t.lemma_ for t in doc]
            bi  = [n.lemma_ for n in ngrams(doc, 2)] 
            tri = [n.lemma_ for n in ngrams(doc, 3)]
        else:
            if self.remove_stops: uni = [t.text for t in doc if not t.is_stop]
            else: uni = [t.text for t in doc]
            bi  = [n.text for n in ngrams(doc, 2)] 
            tri = [n.text for n in ngrams(doc, 3)]

        freq, total = collections.defaultdict(int), 0
        grams = uni + bi + tri
        for tok in grams:
            total += 1
            if tok in freq: freq[tok] += 1
            else: freq[tok] = 1
        
        return [(ID, k,v,(v/total)) for k,v in freq.items() if v >= self.min_word_freq]
    
    def generate_group_norm_df(self):
        """
        Calls `create_group_norms` on each narrative in the dataframe
        """
        freq_series = self.df.dropna(subset='Narrative').apply(lambda x: self.create_group_norms(x['index'], x['Narrative']), axis=1)
        self.freq_series = freq_series
        group_norms = pd.DataFrame([item for sublist in list(freq_series) for item in sublist])
        group_norms = group_norms.rename(columns={
            0:'group_id',
            1:'feat',
            2:'value',
            3:'group_norm'
        })
        self.gn = group_norms
        return group_norms
    
    def get_outcomes(self):
        """
        Pulls the outcome columns from the dataframe and arranges
        them into the correct format
        """
        outcomes = self.df[['ID', 'DASS.Total']]
        outcome_pmn = {}
        for i in range(len(outcomes)):
            outcome_pmn[outcomes.iloc[i].ID] = outcomes.iloc[i]['DASS.Total']
        out = np.array([v for k,v in outcome_pmn.items()])
        self.out = out
        return out
    
    def conf_interval(self, r, samp_size):
        """
        Calculates two sided confidence interval for pearson correlation coefficient (r)
        """
        tup = ()
        if not np.isnan(r):
            z = np.arctanh(r)
            sigma = (1/((samp_size-3)**0.5))
            cint = z + np.array([-1, 1]) * sigma * norm.ppf((1+(1-self.alpha))/2)
            tup = tuple(np.tanh(cint))
        else:
            tup = (np.nan, np.nan)
        return tup

    def get_correl(self, feat):
        """
        Calculates the correlation coefficient between an outcome and a feature

        This function gets the group norms for a given feature and the outcome 
        for the narrative that feature is present in and arranges them into 
        a single array. It then runs `pearsonr` and `conf_interval` to determine
        the level of correlation and its confidence interval. 
        """
        filtered_gn = self.gn[self.gn['feat'] == feat]
        inter = filtered_gn['group_norm'].to_numpy()
        dataContainer = np.zeros(len(self.df))
        gn_group_id = self.gn['group_id'].to_numpy()
        dataContainer[gn_group_id[filtered_gn.index] - 1] = inter
        tup = pearsonr(dataContainer, self.out) + (len(dataContainer),)
        conf = self.conf_interval(tup[0], tup[2])
        return tup + (conf,) + (feat,)           

    def get_correls(self):
        """
        Apply the `get_correl` function to each feature
        """
        correls = self.gn.feat.progress_apply(lambda x: self.get_correl(x))
        self.correls = correls
        return correls

    def dla(self):
        """
        Base function for running all of the above functions sequentially
        """
        print('Generating group norms')
        self.generate_group_norm_df()
        print('Getting outcomes')
        self.get_outcomes()
        print('Correlating text to outcomes')
        correls = self.get_correls()
        return correls

    def save_correls(self, file_path):
        """
        Saves group norms as a CSV (for processing in R)
        """
        correl = pd.DataFrame(self.correls)# .to_csv(file_path)
        correl['pearson_r'] = correl['feat'].apply(lambda x: x[0])
        correl['p_value'] = correl['feat'].apply(lambda x: x[1])
        correl['data_length'] = correl['feat'].apply(lambda x: x[2])
        correl['conf_interval'] = correl['feat'].apply(lambda x: x[3])
        correl['feature'] = correl['feat'].apply(lambda x: x[4])
        correl = correl[['pearson_r', 'p_value', 'conf_interval', 'feature', 'data_length']]
        correl.to_csv(file_path, index=False)

    def load_correls(self, path):
        """
        Loads correl CSV
        """
        correls = pd.read_csv(path)
        correls['conf_interval'] = correls['conf_interval'].apply(ast.literal_eval) 
        correls['feat'] = correls.apply(lambda x: (x['pearson_r'], x['p_value'], x['data_length'], x['conf_interval'], x['feature']), axis=1)
        self.correls = correls.feat

    def save_freq(self, file_path):
        freq_dict = collections.defaultdict(int)
        for dictionary in self.freq_series:
            for _, key, value, _ in dictionary:
                freq_dict[key] += value
        freq_dict = dict(freq_dict)
        df = pd.DataFrame.from_dict(freq_dict, orient='index').reset_index().rename(columns={'index':'feature', 0:'frequency'})
        df.to_csv(file_path, index=False)
    
    def apply_threshold(self, return_words=False):
        """
        Applies the 95% confidence interval by looping through the correlation values
        """
        
        pos = {}
        neg = {}
        for c in self.correls:
            if (c[-1] in pos) and (c[1] < .05) and (c[-2][0] > 0) and (c[-1] not in punctuation):
                pos[c[-1]] = (pos[c[-1]] + c[0])/2
            elif (c[-1] not in pos) and (c[1] < .05) and (c[-2][0] > 0) and (c[-1] not in punctuation):
                pos[c[-1]] = c[0]

        for c in self.correls:
            if (c[-1] in neg) and (c[1] < .05) and (c[-2][0] < 0) and (c[-1] not in punctuation):
                neg[c[-1]] = (neg[c[-1]] + c[0])/2
            elif (c[-1] not in neg) and (c[1] < .05) and (c[-2][0] < 0) and (c[-1] not in punctuation):
                neg[c[-1]] = c[0]
        
        # pos = dict([(c[-1], (c[0], c[1])) for c in self.correls if (c[1] < .05) and (c[-2][0] > 0) and (c[-1] not in punctuation)])
        # neg = dict([(c[-1], (c[0], c[1])) for c in self.correls if (c[1] < .05) and (c[-2][0] < 0) and (c[-1] not in punctuation)]) 

        self.pos = pos
        self.neg = neg

        if return_words:
            return sorted(pos.items(), key=lambda x: x[1], reverse=True), sorted(neg.items(), key=lambda x: x[1], reverse=True)

    def generate_word_clouds(self, path=None, use_name=False):
        """
        Creates wordclouds based on statistically signficant correlation values

        This function will run `apply_threshold` automatically if it hasn't been 
        run before. It will also save the wordclouds at a given path if provided.
        """
        if 'pos' not in self.__dict__: self.apply_threshold()

        p_wc = dict([(p[0], p[1]) for p in self.pos.items()])
        n_wc = dict([(p[0], p[1]) for p in self.neg.items()])

        wc_pos = WordCloud(background_color="white", max_words=1000,)
        wc_pos.generate_from_frequencies(p_wc)
        wc_neg = WordCloud(background_color="white", max_words=1000,)
        wc_neg.generate_from_frequencies(n_wc)
        f, axarr = plt.subplots(2,1)
        
        axarr[0].imshow(wc_pos, interpolation="bilinear")
        axarr[0].axis("off")
        axarr[1].imshow(wc_neg, interpolation="bilinear")
        axarr[1].axis("off")
        plt.show()

        if path:
            path_list = path.split('/')
            for p in path_list:
                os.makedirs(p, exist_ok=True)
            if use_name:
                wc_pos.to_file(f'{path}/{self.name}_pos.png') 
                wc_neg.to_file(f'{path}/{self.name}_neg.png')
            else:            
                wc_pos.to_file(f'{path}/pos.png') 
                wc_neg.to_file(f'{path}/neg.png')

    def search(self, ngram, nb=False):
        """
        Takes in an ngram and displays narratives with that ngram
        in a reading interface
        """
        self.df = self.df.dropna(subset='Narrative')
        excerpt = self.df.loc[self.df.Narrative.str.contains(ngram)]
        if not nb:
            return excerpt
        else:
            def display_text(dropdown_value):
                output.clear_output()
                with output:
                    text = excerpt.loc[excerpt['Abby_Project_ID'] == dropdown_value, 'Narrative'].values[0]
                    wrapped_text = '\n'.join(textwrap.wrap(text, width=80))
                    print(wrapped_text)

            dropdown = widgets.Dropdown(options=excerpt['Abby_Project_ID'].values)
            output = widgets.Output()

            widgets.interactive(display_text, dropdown_value=dropdown)
            display(widgets.VBox([dropdown, output]))

    def regress(self, emb_model='all-MiniLM-L6-v2', device='cpu', kfold=None):
        """
        Embeds narrative column in n-dimensional space using any HuggingFace model,
        then uses features extracted from embeddings to run a RidgeCV regression between
        embeddings and outcomes

        Can ask KFold regression

        Embedding can be run on GPU
        """

        if emb_model.startswith('BAAI'):
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=emb_model, 
                model_kwargs={'device':device}, 
                encode_kwargs={'normalize_embeddings': False}
                )
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name=emb_model, 
                model_kwargs={'device':device}, 
                encode_kwargs={'normalize_embeddings': False}
                )

        embs = np.array(embeddings.embed_documents(list(self.df.Narrative)))
        outs = self.df['DASS.Total'].to_numpy()
        alphas = np.array([1.e+03, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+04, 1.e+05])
        rcv = RidgeCV(alphas=alphas)

        if kfold:
            kfold = KFold(n_splits=kfold)
            for train_index, val_index in kfold.split(embs):
                X_train, X_val = embs[train_index], embs[val_index]
                y_train, y_val = outs[train_index], outs[val_index]

                rcv.fit(X_train, y_train)
                mse = mean_squared_error(y_val, rcv.predict(X_val))
                print(f"Mean squared error: {mse}")
        else:
            rcv.fit(embs, outs)

        self.regscore = rcv.score(embs, outs)
        self.rcv = rcv
        return rcv
