import random
import sys

def shuffle_lines(doc):

    if doc[-1] != '\n':
        doc += '\n'
    if doc[-2] != '\n':
        doc += '\n'

    # Split the string sentences
    sentence_anchors = [pos for pos, char in enumerate(doc) if char == '\n']
    sentences = []
    for i in range(len(sentence_anchors)):
        if i == 0:
            anchor_pre = 0
        else:
            anchor_pre = sentence_anchors[i - 1] + 1
        anchor = sentence_anchors[i]
        if anchor_pre == anchor:
            sentences.append(doc[anchor])
        else:
            sentences.append(doc[anchor_pre:anchor])

    # Split the sentences into blocks of sentences
    block_anchors = []
    block_stop = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if sentence[:9] == 'Question:':
            block_anchors.append(i)
        if sentence[0] == '\n':
            block_stop.append(i)
    if len(block_anchors) != len(block_stop):
        sys.exit('Block index error!')

    # Shuffle blocks
    idx_shuffled = []
    for i in range(len(block_anchors)):
        idx_shuffled.append(block_anchors[i])
        idx_block = [item for item in range(block_anchors[i] + 1, block_stop[i])]
        random.shuffle(idx_block)
        idx_shuffled += idx_block
        idx_shuffled.append(block_stop[i])

    # Rearrange the sentences
    doc = ""
    for i in range(len(idx_shuffled)):
        doc += sentences[idx_shuffled[i]] + '\n'

    return doc


def shuffle_words(doc):

    if doc[-1] != '\n':
        doc += '\n'
    if doc[-2] != '\n':
        doc += '\n'

    # Split the string sentences
    sentence_anchors = [pos for pos, char in enumerate(doc) if char == '\n']
    sentences = []
    for i in range(len(sentence_anchors)):
        if i == 0:
            anchor_pre = 0
        else:
            anchor_pre = sentence_anchors[i - 1] + 1
        anchor = sentence_anchors[i]
        if anchor_pre == anchor:
            sentences.append(doc[anchor])
        else:
            sentences.append(doc[anchor_pre:anchor])

    # Split the sentences into blocks of sentences
    block_anchors = []
    block_stop = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if sentence[:9] == 'Question:':
            block_anchors.append(i)
        if sentence[0] == '\n':
            block_stop.append(i-1)
    if len(block_anchors) != len(block_stop):
        sys.exit('Block index error!')

    # Shuffle words in blocks
    doc = ""
    for i in range(len(block_anchors)):
        doc += sentences[block_anchors[i]] + '\n'
        doc_block = ""
        for j in range(block_anchors[i] + 1, block_stop[i]):
            doc_block += sentences[j]
        # shuffle words
        words = doc_block.split()
        random.shuffle(words)
        doc_block_shuffled = ""
        for word in words:
            doc_block_shuffled += word
            doc_block_shuffled += ' '
        doc += doc_block_shuffled + '\n'
        doc += sentences[block_stop[i]] + '\n' + '\n'

    return doc

def defense(prompt, strategy=None):
    if strategy != None:
        if strategy == "basic":
            prompt_shuffled = shuffle_lines(prompt)
            
        elif strategy == "super":
            prompt_shuffled = shuffle_words(prompt)
    else:
        prompt_shuffled = prompt
    return prompt_shuffled