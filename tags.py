# Ashley Morales Project II
from operator import itemgetter
import numpy as np
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
from gensim.parsing.preprocessing import remove_stopwords
from collections import Counter
from math import sqrt

def word2vec(word):
    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c * c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch] * v2[0][ch] for ch in common) / v1[2] / v2[2]

def get_bert_based_similarity(termA, termB, model, tokenizer, tag):
    termA = termA.replace(tag, "").lower()
    termA = remove_stopwords(termA)
    termB = termB.replace(tag, "").lower()
    termB = remove_stopwords(termB)
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
dict_of_tags = {} # term: conceptID
dict_of_attributes = {} # conceptID: number of attributes

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

def descriptionFile(active_conceptIDs,tag):
    global dict_of_tags
    with open("sct2_Description_Snapshot-en_INT_20210731.txt", "r") as d:
        for line in d:
            temp = line.split("\t")
            active = temp[2]  # should equal "1"
            conceptID = temp[4]
            typeID = temp[6]  # should equal "900000000000003001" for FSN
            term = temp[7]

            # active concepts should also have active description
            # term with specified tag should only be an FSN, denoted by typeID
            # if all true, then append term to a dictionary with its corresponding conceptID

            if active == "1" and typeID == "900000000000003001" and tag in term:
                if conceptID in active_conceptIDs:
                    if term not in dict_of_tags:
                        dict_of_tags[term] = conceptID

def relationshipFile():
    with open("sct2_Relationship_Snapshot_INT_20210731.txt", "r") as df:
        for line in df:
            temp = line.split("\t")
            active = temp[2]  # should equal "1"
            sourceID = temp[4]
            typeID = temp[7]  # non-hierarchical equal "116680003"

            # count number of attributes (rows) for all active concepts
            # where typeID != "116680003" (non-hierarchical relation)
            # append conceptId and number of attributes to dictionary

            if active == "1" and typeID != "116680003":
                if sourceID in dict_of_attributes:
                    dict_of_attributes[sourceID] += 1
                else:
                    dict_of_attributes[sourceID] = 1

        # if active concepts did not have any non-hierarchical relations
        # then number of attributes is 0
        for i in active_conceptIDs:
            if i not in dict_of_attributes:
                dict_of_attributes[i] = 0

def CompareandWrite(termA, tag):
    global dict_of_tags
    global dict_of_attributes
    list_of_similarities = []
    fileName = termA.replace(tag, "") + ".txt"
    CID1 = dict_of_tags[termA]
    # Lexical similarity part
    # convert termA to word2vec for finding cosine distance
    termA_clean = termA.replace(tag, "").lower()
    termA_clean = remove_stopwords(termA_clean)
    w1 = word2vec(termA_clean)
    # create list of active terms
    list_of_concepts = list(dict_of_tags.keys())

    # loop through list to compare termA with every termB in list
    for termB in list_of_concepts:
        termB_clean = termB.replace(tag, "").lower()
        termB_clean = remove_stopwords(termB_clean)
        w2 = word2vec(termB_clean)
        sim = cosdis(w1, w2)
        print(termA, ",", termB, "")
        print(sim, "\n")
        # create list with cos distance and pair of terms
        list_of_similarities.append([sim, termA, termB])

    print("\nNow showing top 10 similar concepts to:", termA)

    # sort list by its cosine distance in descending order so higher elements on the list indicate most similar terms
    list_of_similarities = sorted(list_of_similarities, key=itemgetter(0), reverse=True)

    file = open(fileName, "w")
    file.write("CID" + "\t" + "FSN\n")
    scores = []
    # Semantic similarity part
    # iterate through first 10 terms in list and apply the pubmed model for each
    for i in range(1, 10):
        score1 = get_bert_based_similarity(list_of_similarities[i][1], list_of_similarities[i][2], pubmed_bert_model, pubmed_bert_tokenizer, tag)
        score2 = get_bert_based_similarity(list_of_similarities[i][1], list_of_similarities[i][2], coder_model, coder_tokenizer, tag)
        similarity_score = (score1 * 0.6) + (score2 * 0.4)
        # print(similarity_score, "\t", list_of_similarities[i][2])
        scores.append([similarity_score, list_of_similarities[i][2]])

        CID2 = dict_of_tags[list_of_similarities[i][2]]

        # write concept pair to file if similarity is above 90% and they have different number of attributes
        if similarity_score >= 0.90 and dict_of_attributes[CID1] != dict_of_attributes[CID2]:
            file.write(CID1 + '\t' + list_of_similarities[i][1] + '\n' + CID2 + '\t' + list_of_similarities[i][2] + '\n')
            file.write('Score: ' + str(similarity_score) + '\n\n')
    file.close()
    scores = sorted(scores, key=itemgetter(0), reverse=True)
    for x in scores:
        print(x)

if __name__=='__main__':
    # tag = " (disorder)"
    # tag = " (finding)"
    tag = input("Enter tag: ")
    print("Creating list of all active concepts...")
    conceptFile()
    print("Done!\n")

    print(f"Creating dictionary of active concepts with tag '{tag}' and their concept IDs...")
    descriptionFile(active_conceptIDs, tag)
    print("Done!\n")

    print("Creating dictionary of attributes for each concept...")
    relationshipFile()
    print("Done!\n")

    # term1 = "Specimen from urinary bladder (specimen) cid=450872001"

    termA = input("Enter term to compare: ")
    CompareandWrite(termA, tag)
