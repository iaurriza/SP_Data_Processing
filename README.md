# SP_Data_Processing

## Parts

    Cleaner.py
        - input:    /need_cleaning
        - output:   /cleaned
        - function: replace contractions
    InputPipeline.py
        - imports:  split_module.py
                    spellcheck_module.py
                    suggestions_cleared.csv
        - function: makes a pipline that intakes a single raw text and outputs cleaned text splitted into several sentences

    vectorizer.py
        - input:    /cleaned_sorted
         First Classifier. Input consists of a single sentence with a polarity of a single label (i.e. Audio, Graphics, Gameplay).

    multilabel_vectorizer.py
        - input:    /cleaned
         Second Classifier. Input consists of a single sentence with label type and a polarity for each type. 

    organizer_2.py
        - input:    /cleaned
        - output:   /cleaned_sorted
