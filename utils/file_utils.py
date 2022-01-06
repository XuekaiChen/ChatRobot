# -*- coding: utf-8 -*-
# Date        : Jan-01-06
# Author      : Chen Xuekai
# Description : file utils for directly reading files of the dataset.


from __future__ import absolute_import  # 区分相对import和绝对import重名
from __future__ import division  # 精确除法（python3默认有）
from __future__ import print_function  # 在python2中可以使用python3的print
from __future__ import unicode_literals

import re
from io import open


def print_lines(file, n=10):
    """open a text file and load its lines, print its first n lines.
    """
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


def load_lines(filename, fields):
    # Splits each line of the file into a dictionary of fields
    lines = {}
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]
            lines[line_obj['lineID']] = line_obj
    return lines


def load_conversations(filename, lines, fields):
    # Groups fields of lines from `load_lines` into conversations based on *movie_conversations.txt*
    conversations = []
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


def extract_sentence_pairs(conversations):
    # Extracts pairs of sentences from conversations
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        # We ignore the last line (no answer for it)
        for i in range(len(conversation["lines"]) - 1):
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs
