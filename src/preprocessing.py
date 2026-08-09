import re
import unicodedata

NER_REPLA = {
    "PERSON": "_PERSON_",
    "DATE": "_DATE_",
    "TIME": "_TIME_",
    "CARDINAL": "_NUMBER_",
    "ORDINAL": "_ORDINAL_",
}

KEEP_STOPWORDS = {"no", "not", "never", "without", "against"}
ALLOWED_POS = {"NOUN", "PROPN", "VERB", "ADJ"}


def load_spacy_model():
    import spacy
    return spacy.load(
        "en_core_web_sm",
        enable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "ner", "parser"],
    )


def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"::.*$", "", text).strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str, nlp) -> str:
    """Return space-joined lemmas using the notebook NLP pipeline."""
    text = normalize_text(text)
    doc = nlp(text)
    lemmas = []
    i = 0
    while i < len(doc):
        token = doc[i]

        if token.is_space or token.like_url or token.like_email:
            i += 1
            continue

        if token.ent_iob_ == "B":
            ent_type = token.ent_type_
            if ent_type in NER_REPLA:
                lemmas.append(NER_REPLA[ent_type])
                i += 1
                while i < len(doc) and doc[i].ent_iob_ == "I":
                    i += 1
                continue

        if token.ent_iob_ == "I":
            i += 1
            continue

        if token.is_punct:
            i += 1
            continue

        if token.is_stop and token.lower_ not in KEEP_STOPWORDS:
            i += 1
            continue

        if token.pos_ not in ALLOWED_POS:
            i += 1
            continue

        lemma = token.lemma_.strip().lower()
        if lemma == "-pron-":
            lemma = token.lower_

        if lemma.upper() in NER_REPLA.values():
            lemmas.append(lemma.upper())
            i += 1
            continue

        if re.fullmatch(r"[a-zA-Z]+(?:-[a-zA-Z]+)*", lemma):
            lemmas.append(lemma)

        i += 1

    return " ".join(lemmas)
