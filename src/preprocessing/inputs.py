from pathlib import Path
from typing import List, Tuple
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

import pandas as pd


def df_cleanup(df: pd.DataFrame, lemmatize=True, lower=True):
    reg_exp_tokenizer = RegexpTokenizer(r"\w+")
    word_lemmatizer = WordNetLemmatizer()

    def normalize_text(x):
        if lower:
            x = x.lower()
        if lemmatize:
            tokens = [word_lemmatizer.lemmatize(word) for word in reg_exp_tokenizer.tokenize(x)]
        else:
            tokens = [word for word in reg_exp_tokenizer.tokenize(x)]
        return " ".join(tokens)

    # drop strange rows
    df.drop(df[df["Season"] == "Season"].index, inplace=True)
    # drop all rows with nans
    df.dropna(inplace=True)
    # cast fixed columns to int
    df["Season"] = df["Season"].astype(int)
    df["Episode"] = df["Episode"].astype(int)
    # remove blanks from lines
    df["Line"] = df["Line"].str.strip()
    # normalize text
    df["Line"] = df["Line"].apply(normalize_text)
    # adding new target
    df["Addressee"] = df["Character"].shift(periods=-1)


def classifier_input(df: pd.DataFrame, characters: List[str], target: str = "Addressee") -> List[Tuple[str, str]]:
    """
    Returns list of the following format for filtered characters
    [
        (line, target)
    ]

    Example:
    [
        ("I'm gonna miss him.  I'm gonna miss Chef and I...and I don't know how to tell him!", 'Stan'),
        ('Reverse to you, Jew.', 'Stan'),
        ('All right!', 'Kyle')
    ]
    """
    return [(i["Line"], i[target]) for _, i in df[df["Character"].isin(characters)].iterrows()]


def generator_input(df: pd.DataFrame, characters: List[str], n_context: int = 1):
    """
    Returns dict of the following format
    {
        character_1: [
                (line1, [context_line_1_to_line_1, context_character_1_to_line_1], [context_line_2_to_line_1, context_character_2_to_line_1], ...),
                (line2, [context_line_1_to_line_2, context_character_1_to_line_2], [context_line_2_to_line_2, context_character_2_to_line_2], ...),
                ...
        ],
        ...
    }

    Example:
    {
        "Cartman": [
            ("I'm gonna miss him.  I'm gonna miss Chef and I...and I don't know how to tell him!", ["I hope you're making the right choice.", 'Mrs. Garrison'], ["What's the meaning of life? Why are we here?", 'Chef']),
            ('Reverse to you, Jew.', ['Draw two card, fatass.', 'Kyle'], ['Good-bye! ..', 'Chef']),
            ...
        ],
        ...
    }
    """
    res = {}
    contexts = [df[["Line", "Character"]].shift(i) for i in range(1, n_context + 1)]
    for ch in characters:
        mask = df["Character"] == ch
        res[ch] = list(zip(df[mask]["Line"].tolist(), *[c[mask].values.tolist() for c in contexts]))
    return res


if __name__ == "__main__":
    # Example of code usage
    df = pd.read_csv(Path("../../SouthParkData/All-seasons.csv"))
    df_cleanup(df)
    gen_input = generator_input(df, ["Cartman"], n_context=2)

    print(gen_input["Cartman"][0])
    print(gen_input["Cartman"][1])

    class_input = classifier_input(df, ["Cartman"])

    print(class_input[:3])
