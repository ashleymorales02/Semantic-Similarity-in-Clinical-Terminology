# Ashley Morales Project II
import numpy as np
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
from gensim.parsing.preprocessing import remove_stopwords
import re

def get_bert_based_similarity(termA, termB, model, tokenizer):
    termA = re.sub('\([^()]*\)', '', termA)
    termA = remove_stopwords(termA).lower()
    termB = re.sub('\([^()]*\)', '', termB)
    termB = remove_stopwords(termB).lower()
    """
    computes the embeddings of termA and its similarity with each corresponding termB
    Args:
        model: the language model
        tokenizer: the tokenizer to consider for the computation
    Returns:
        similaritiy: similarity measure of termA and termB between 0-1
    """
    # Transform input tokens
    inputs_1 = tokenizer(termA, return_tensors='pt')
    # print(inputs_1)
    inputs_2 = tokenizer(termB, return_tensors='pt')
    # print(inputs_2)
    sent_1_embed = np.mean(model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)
    # print(sent_1_embed)
    sent_2_embed = np.mean(model(**inputs_2).last_hidden_state[0].detach().numpy(), axis=0)
    # print(sent_2_embed)
    similarity = dot(sent_1_embed, sent_2_embed) / (norm(sent_1_embed) * norm(sent_2_embed))
    return similarity

# Download pytorch model
pubmed_bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
coder_model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
coder_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')

active_conceptIDs = set()
dict_of_concepts = {}  # conceptID:term

def conceptFile():
    global active_conceptIDs
    with open("sct2_Concept_Snapshot_INT_20210731.txt", "r") as f:
        for line in f:
            temp = line.split("\t")
            id = temp[0]
            active = temp[2]  # should equal "1"

            # put all active concepts into list
            if active == "1":
                active_conceptIDs.add(id)

def Concepts():
    global dict_of_concepts
    with open("sct2_Description_Snapshot-en_INT_20210731.txt", "r") as d:
        for line in d:
            temp = line.split("\t")
            active = temp[2]  # should equal "1"
            conceptID = temp[4]
            typeID = temp[6]
            term = temp[7]

            if active == "1" and typeID == "900000000000003001" and conceptID in active_conceptIDs:
                    if term not in dict_of_concepts:
                        dict_of_concepts[conceptID] = term

def FindConcept(ID):
    global dict_of_concepts
    for concept in dict_of_concepts.keys():
        if concept == ID:
            return dict_of_concepts[ID]

def getSims(termA):
    termA_clean = re.sub('\([^()]*\)', '', termA)
    termA_clean = remove_stopwords(termA_clean)

    list_of_concepts = set(dict_of_concepts.values())

    words = termA_clean.split()
    for termB in list_of_concepts:
        if termA != termB:
            if all(ext in termB for ext in words):
                score1 = get_bert_based_similarity(termA, termB, pubmed_bert_model, pubmed_bert_tokenizer)
                score2 = get_bert_based_similarity(termA, termB, coder_model, coder_tokenizer)
                similarity_score = (score1 * 0.6) + (score2 * 0.4)
                print(similarity_score, "\t", termB)

if __name__=='__main__':
    conceptFile()
    Concepts()
    cid = input("Enter concept id: ")
    termA = FindConcept(cid)
    print("The corresponding term is: ", termA, "\n")

    print(f"Finding similar concepts to {termA}...")
    getSims(termA)

