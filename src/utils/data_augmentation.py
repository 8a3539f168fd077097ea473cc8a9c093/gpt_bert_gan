import random

from utils import text_process
import csv


def simple_sentences_rotation(dataset_path, dataset_dest, keep_original=True,
                              uniform_rotation=False, cut_a=False):
    """Create a new dataset based on another one with rotated sentences"""
    rotated_sentences = []

    # get all sentences of the dataset in a list
    original_sentences = text_process.get_tokenlized(dataset_path)

    for sentence in original_sentences:
        for i in range(5):
            if sentence[-1] != '.':
                rotated_sentence = sentence[i + 1:] + sentence[:i + 1]
            else:
                rotated_sentence = sentence[i + 1:len(sentence) - 2] + sentence[:i + 1] + list('.')
            if cut_a:
                if rotated_sentence[0] == 'a':
                    rotated_sentence = rotated_sentence[1:]
            rotated_sentences.append(rotated_sentence)

        # if uniform_rotation true, we do a random rotation
        if uniform_rotation:
            random_index = random.randint(1, 32)
            if sentence[-1] != '.':
                rotated_sentence = sentence[random_index + 1:] + sentence[:random_index + 1]
            else:
                rotated_sentence = sentence[random_index + 1:len(sentence) - 2] + sentence[:random_index + 1] + list(
                    '.')
            if cut_a:
                if rotated_sentence[0] == 'a':
                    rotated_sentence = rotated_sentence[1:]
            rotated_sentences.append(rotated_sentence)

    # Write original and new sentences into a file
    if keep_original:
        text_process.write_tokens(dataset_dest, original_sentences)
    text_process.write_tokens(dataset_dest, rotated_sentences)


def cut_first_token(dataset_path, dataset_dest):
    """Modfiy the dataset such that the sentence never start with a given token"""

    cut_sentences = []

    # get all sentences of the dataset in a list
    original_sentences = text_process.get_tokenlized(dataset_path)

    for sentence in original_sentences:
        if sentence[0] == 'a':
            cut_sentence = sentence[1:]
        else:
            cut_sentence = sentence
        cut_sentences.append(cut_sentence)

    # Write original and new sentences into a file
    text_process.write_tokens(dataset_dest, cut_sentences)


def reduce_dataset(dataset_path, dataset_dest):
    kept_sentences = []

    # get all sentences of the dataset in a list
    original_sentences = text_process.get_tokenlized(dataset_path)
    for i in range(10000):
        kept_sentences.append(original_sentences[i])
    text_process.write_tokens(dataset_dest, kept_sentences)


def create_fake_true_dataset(fake_data_path, true_data_path):
    """
    Creates dataset for the gpt_bert GAN from true and fake data labeled.
    """
    fake_sentences = text_process.get_tokenlized(fake_data_path)
    true_sentences = text_process.get_tokenlized(true_data_path)

    header = ['text', 'label']
    data = []
    for sentence in fake_sentences:
        # fake has the 0 label and true data has 1 label
        data.append([sentence, 0])

    for sentence in true_sentences:
        data.append([sentence, 1])

    with open('image_coco_fake_true.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


# The idea will be to rename the new file to image_coco so that no code has to be modified
# simple_sentences_rotation("image_coco.txt", "image_coco_with_rotations.txt")
# simple_sentences_rotation("image_coco.txt", "image_coco_with_random_rotations.txt",
#                          keep_original=False, uniform_rotation=False, cut_a=True)
# cut_first_token("image_coco.txt", "image_coco_with_no_a")
#reduce_dataset("emnlp_news.txt", "emnlp_news_tiny.txt")