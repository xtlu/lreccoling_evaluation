This open-sourced dataset contains crowdsourced annotations for two text classification datasets: IMDB (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and AGNEWS (https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset). 
Both datasets are in the format of [Text Label Saliency Words Correct]
Text: Top important words shown for workers. All punctuation and special tokens were ignored.
Label: 0(negative), 1(positive) in IMDB and 0(World), 1(Sports), 2(Business), 3(Science) in AGNEWS.
Saliency: Either of All_Attention, Last_Attention, Vanilla_Gradient, InputXGrad, Integrated_Gradient, DeepLIFT, LIME, Random.
Words: Number of showing words.
Correct: Whether or not workers considered the label of showing text was the same as the ground-truth label. 0(not correct) and 1(correct) by majority voting.