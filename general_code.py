import threading

import nltk
import re
import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import det_curve
import matplotlib.pyplot as plt
import glob
from itertools import combinations
from sklearn.metrics import det_curve


def readFile(filepath):
    with open(filepath, "r") as df:
        content = df.read()
    return content



def preprocess_text(text):
    text_lowercase = text.lower()
    # Remove special characters
    charaters_to_remove = ['"', "'", '  ', ',', '?', '!']
    for c in charaters_to_remove:
        text_lowercase = text_lowercase.replace(c, '')

    # Remove section breaks
    section_break_pattern = r"\f"
    cleaned_text = re.sub(section_break_pattern, '', text_lowercase)
    return cleaned_text

def text_to_ngrams_char_level(text, n):
    ngrams = list(nltk.ngrams(text, n))
    ngrams = [''.join(ngram) for ngram in ngrams]
    ngrams = [ngram.replace(' ', '_') for ngram in ngrams]
    return ngrams

def text_to_ngrams_word_level(text, n):
    tokens = nltk.word_tokenize(text)
    ngrams = list(nltk.ngrams(tokens, n))
    ngrams = [' '.join(ngram) for ngram in ngrams]
    return ngrams


def ngrams_dict_with_order(l):
    l_ngrams = {}
    for elmt in l:
        if elmt not in l_ngrams.keys():
            l_ngrams[elmt] = 1
        else:
            l_ngrams[elmt] += 1
    for item in l_ngrams:
        l_ngrams[item] = l_ngrams[item] / len(l_ngrams)
    return l_ngrams


def dicts_intersection(d1, d2):
    d1 = {k: v for k, v in d1.items() if k in d2}
    d2 = {k: v for k, v in d2.items() if k in d1}
    return d1, d2



def A_distance_from_dicts(d1, d2, inner_treshold_1, inner_threshold_2):
    number_of_shared_ngrams = len(d1)
    number_of_similar_ngrams = 0
    for item in d1:
        score_of_similarity = max(d1[item], d2[item]) / min(d1[item], d2[item])
        if score_of_similarity >= inner_treshold_1 and score_of_similarity <= inner_threshold_2 :
            number_of_similar_ngrams += 1
        elif score_of_similarity < inner_treshold_1:
            number_of_similar_ngrams += 2
    try:
        return number_of_similar_ngrams / number_of_shared_ngrams
    except:
        return number_of_similar_ngrams / 0.1



def A_distance(reference_texts_list, test_text, inner_threshold_1, inner_threshold_2, ngram_size, level='char'): # ngram_size = 2, 3, 4 etc. | level = 'word' or 'char'
    ref_text = ""
    for t in reference_texts_list:
        ref_text += t

    ref_text = preprocess_text(ref_text)
    test_text = preprocess_text(test_text)

    if level == 'word':
        ref_text_ngrams = text_to_ngrams_word_level(ref_text, ngram_size)
        test_text_ngrams = text_to_ngrams_word_level(test_text, ngram_size)

    else:
        ref_text_ngrams = text_to_ngrams_char_level(ref_text, ngram_size)
        test_text_ngrams = text_to_ngrams_char_level(test_text, ngram_size)

    ngrams_ref_text, ngrams_test_text = dicts_intersection(ngrams_dict_with_order(ref_text_ngrams), ngrams_dict_with_order(test_text_ngrams))

    return A_distance_from_dicts(ngrams_ref_text, ngrams_test_text, inner_threshold_1, inner_threshold_2)







def plot_det_curve(genuine_scores_list, imposter_scores_list, label):
    # Compute False Positive Rate (FPR) and True Positive Rate (TPR) at different thresholds
    scores = np.concatenate((genuine_scores_list, imposter_scores_list))
    labels = np.concatenate((np.ones(len(genuine_scores_list)), np.zeros(len(imposter_scores_list))))
    fpr, fnr, thresholds = det_curve(labels, scores)

    # Compute the Equal Error Rate (EER)
    eer_threshold = np.argmin(np.abs(fpr - (1 - fnr)))
    eer = (fpr[eer_threshold] + (1 - fnr[eer_threshold])) / 2.0

    # Plot the DET curve and EER point
    plt.plot(fpr, fnr, label=label)
    plt.plot([eer], [1 - eer], marker='o', markersize=5, label='EER = {:.3f}'.format(eer))
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.legend()
    plt.show()



# DET curve plotting for mutiple pair of input (genuine_scores_list, imposter_scores_list)
def plot_det_mutiple_curves(genuine_scores_list, imposter_scores_list, legends_list, title, filename):
    if len(genuine_scores_list) != len(imposter_scores_list):
        print("Different lengths for the input lists")
        return
    N = len(genuine_scores_list)
    fpr_list = []
    fnr_list = []
    eer_list = []
    for i in range(N):
        genuine_scores = genuine_scores_list[i]
        imposter_scores = imposter_scores_list[i]
        all_scores = np.concatenate((genuine_scores, imposter_scores))
        labels = np.concatenate((np.ones_like(genuine_scores), np.zeros_like(imposter_scores)))
        fpr, fnr, thresholds = det_curve(labels, all_scores)
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        print("Length of fpr_list:", len(fpr_list))
        print("Length of fnr_list:", len(fnr_list))
        # Compute the Equal Error Rate (EER)
        eer_threshold = np.argmin(np.abs(fpr - (1 - fnr)))
        eer = (fpr[eer_threshold] + (1 - fnr[eer_threshold])) / 2.0
        eer_list.append(eer)

    # plot DET curves
    plt.figure()
    for i in range(N):
        plt.plot(fpr_list[i], fnr_list[i], label=f'{legends_list[i]}')
    for i in range(N):
        eer_idx = np.argmin(np.abs(np.array(fpr_list[i]) - np.array(fnr_list[i])))
        plt.plot(fpr_list[i][eer_idx], fnr_list[i][eer_idx], 'o', color='black', markersize=4, label=legends_list[-1])

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('False Negative Rate (FNR)')
    plt.title(title)
    plt.grid(True)
    plt.legend(legends_list)
    plt.savefig(f'/home/youssef/Desktop/A-measure-advanced/A_measure_advanced_figures/normal/{filename}')








if __name__ == "__main__":
    inner_threshold_values = [[1.05, 1.1], [1.1, 1.25], [1.25, 1.5], [1.5, 1.75], [1.75, 2]]
    K_values = [1, 2, 3, 4, 5, 6]
    ngrams_sizes = [2, 3, 4]
    levels = ['char', 'word']
    root_dir = '/home/youssef/Desktop/ReedSy/DATASETS/Reedsy_Prompts_short_stories'
    authors_dirs = glob.glob(f"{root_dir}/*")
    for level in levels:
        print(f"***** level = {level} *****")
        for ngram_size in ngrams_sizes:
            print(f"    ***** ngram_size = {ngram_size} *****")
            for K in K_values:
                print(f"        ***** K = {K} *****")
                genuine_scores_list = []
                imposter_scores_list = []
                ai_scores_list = []
                for inner_thresholds in inner_threshold_values:
                    print(f"            ***** inner_threshold = {inner_thresholds} *****")
                    genuine_scores = []
                    # genuine scores list filling
                    for author_dir in authors_dirs:
                        txt_files = glob.glob(author_dir + "/*.txt")
                        author_initials = re.sub(r'\d+', '', txt_files[0].split("/")[-1].split(".")[0])
                        ref_texts = []
                        for i in range(1, K+1):
                            ref_texts.append(readFile(f"{author_dir}/{author_initials}{i}.txt"))
                        test_texts = [readFile(txt_file) for txt_file in txt_files if readFile(txt_file) not in ref_texts]

                        for test_text in test_texts:
                            genuine_scores.append(A_distance(reference_texts_list=ref_texts, test_text=test_text, inner_threshold_1=inner_thresholds[0], inner_threshold_2=inner_thresholds[1], ngram_size=ngram_size, level=level))
                    genuine_scores_list.append(genuine_scores)

                    imposter_scores = []
                    # imposter scores list filling
                    for author_dir in authors_dirs:
                        txt_files = glob.glob(author_dir + "/*.txt")
                        author_initials = re.sub(r'\d+', '', txt_files[0].split("/")[-1].split(".")[0])
                        ref_texts = []
                        for i in range(1, K+1):
                            ref_texts.append(readFile(f"{author_dir}/{author_initials}{i}.txt"))
                        test_texts = []
                        for auth_dir in authors_dirs:
                            if auth_dir != author_dir:
                                files = glob.glob(auth_dir + "/*.txt")
                                for f in files:
                                    if int(re.findall(r'\d+', f.split("/")[-1])[0]) > K:
                                        test_texts.append(readFile(f))

                        for test_text in test_texts:
                            imposter_scores.append(A_distance(reference_texts_list=ref_texts, test_text=test_text, inner_threshold_1=inner_thresholds[0], inner_threshold_2=inner_thresholds[1], ngram_size=ngram_size, level=level))

                    ai_scores = []
                    # ai scores list filling
                    for author_dir in authors_dirs:
                        txt_files = glob.glob(author_dir + "/*.txt")
                        author_initials = re.sub(r'\d+', '', txt_files[0].split("/")[-1].split(".")[0])
                        ref_texts = []
                        for i in range(1, K+1):
                            ref_texts.append(readFile(f"{author_dir}/{author_initials}{i}.txt"))

                        # Test of stories written by author on specific topics
                        root_dir = '/home/youssef/Desktop/ReedSy/DATASETS/Reedsy_Prompts_short_stories_AI_Person_style'
                        author_name = author_dir.split("/")[-1]
                        test_files = glob.glob(f"{root_dir}/{author_name}/*.txt")
                        test_texts = [readFile(f) for f in test_files]
                        for test_text in test_texts:
                            ai_scores.append(A_distance(reference_texts_list=ref_texts, test_text=test_text, inner_threshold_1=inner_thresholds[0], inner_threshold_2=inner_thresholds[1], ngram_size=ngram_size, level=level))

                        # Test of stories written by author on random topics
                        root_dir = '/home/youssef/Desktop/ReedSy/DATASETS/Stories_AI_person_style'
                        test_files = glob.glob(f"{root_dir}/{author_name}/*.txt")
                        if len(test_files) > 0:
                            test_texts = [readFile(f) for f in test_files]
                            for test_text in test_texts:
                                ai_scores.append(A_distance(reference_texts_list=ref_texts, test_text=test_text, inner_threshold_1=inner_thresholds[0], inner_threshold_2=inner_thresholds[1], ngram_size=ngram_size, level=level))

                    imposter_scores_list.append(imposter_scores + ai_scores)
                    ai_scores_list.append(ai_scores)

                # DET curve plotting for specific level, ngram_size and K value
                plot_det_mutiple_curves(genuine_scores_list, imposter_scores_list, ["inner_threshold_1 = 1.05\ninner_threshold_2 = 1.1", 'inner_threshold_1 = 1.1\ninner_threshold_2 = 1.25', 'inner_threshold_1 = 1.25\ninner_threshold_2 = 1.5', 'inner_threshold_1 = 1.5\ninner_threshold_2 = 1.75', 'inner_threshold_1 = 1.75\ninner_threshold_2 = 2', 'EER']
                                        , f'DET curve - {level} level - {ngram_size}-grams - K = {K}', f'level_{level}_{ngram_size}_grams_K_{K}')
