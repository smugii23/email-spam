import os
import re
import numpy as np

# Paths to the train and test directories
TRAIN_PATH = "./train"
TEST_PATH = "./test"

def number_of_all_emails():
    """Returns the total number of emails in the train directory."""
    counter = sum(len(files) for _, _, files in os.walk(TRAIN_PATH))
    return counter

def number_of_spam_emails():
    """Returns the number of spam emails in the train directory."""
    counter = sum(1 for _, _, files in os.walk(TRAIN_PATH) for filename in files if "spam" in filename)
    return counter

def number_of_ham_emails():
    """Returns the number of ham emails in the train directory."""
    counter = sum(1 for _, _, files in os.walk(TRAIN_PATH) for filename in files if "ham" in filename)
    return counter

def text_parser(text):
    """Parses the given text and returns a list of words."""
    words = re.split("[^a-zA-Z]", text)
    return [word.lower() for word in words if word]

def generate_train_words():
    """Returns the words of train data: (all_train_words, spam_train_words, ham_train_words)"""
    all_words, spam_words, ham_words = [], [], []

    for root, _, files in os.walk(TRAIN_PATH):
        for filename in files:
            full_path = os.path.join(root, filename)
            with open(full_path) as file:
                words = text_parser(file.read())
                all_words.extend(words)
                if "ham" in filename:
                    ham_words.extend(words)
                elif "spam" in filename:
                    spam_words.extend(words)

    return sorted(all_words), sorted(spam_words), sorted(ham_words)

def unique_words(all_train_words):
    """Returns all unique words from the train data."""
    return sorted(set(all_train_words))

def frequency_calculator(words):
    """Calculates the frequency of given words."""
    return {word: words.count(word) for word in set(words)}

def generate_bag_of_words(all_unique_words, spam_train_words, ham_train_words):
    """Returns the frequency of each word in spam and ham classes: (spam_bag_of_words, ham_bag_of_words)"""
    spam_bag_of_words = frequency_calculator(spam_train_words)
    ham_bag_of_words = frequency_calculator(ham_train_words)

    for word in all_unique_words:
        spam_bag_of_words.setdefault(word, 0)
        ham_bag_of_words.setdefault(word, 0)

    return (dict(sorted(spam_bag_of_words.items())),
            dict(sorted(ham_bag_of_words.items())))

def smoothed_bag_of_words(all_unique_words, spam_bag_of_words, ham_bag_of_words, delta):
    """Returns the smoothed bag of words for spam and ham classes."""
    smoothed_spam_bow = {word: count + delta for word, count in spam_bag_of_words.items()}
    smoothed_ham_bow = {word: count + delta for word, count in ham_bag_of_words.items()}

    return (dict(sorted(smoothed_spam_bow.items())),
            dict(sorted(smoothed_ham_bow.items())))

def class_probability(nb_of_all_emails, nb_of_class_emails):
    """Calculates the probability of a class."""
    return nb_of_class_emails / nb_of_all_emails

def conditional_probability(all_unique_words, bag_of_words, smoothed_bow, delta):
    """Calculates the conditional probability P(Wi|class) for each word."""
    total_words = sum(bag_of_words.values()) + (delta * len(all_unique_words))
    return {word: count / total_words for word, count in smoothed_bow.items()}

def generate_model_output(word_count, words, ham_wf, ham_cp, spam_wf, spam_cp):
    """Generates the content of model.txt."""
    return "\n".join(f"{i+1}  {words[i]}  {ham_wf[words[i]]}  {ham_cp[words[i]]}  {spam_wf[words[i]]}  {spam_cp[words[i]]}" 
                     for i in range(word_count))

def build_model_file(model_output):
    """Writes the model output to model.txt."""
    with open("model.txt", 'w') as model_file:
        model_file.write(model_output)

def get_test_file_names():
    """Returns the names of all test emails."""
    return [filename for _, _, files in os.walk(TEST_PATH) for filename in files]

def get_actual_labels():
    """Returns the actual labels (ham or spam) of each test email."""
    return ["ham" if "ham" in filename else "spam" for _, _, files in os.walk(TEST_PATH) for filename in files]

def calculate_scores(all_unique_words, spam_prob, ham_prob, spam_cond_prob, ham_cond_prob, delta):
    """Calculates the scores for ham and spam for each test email."""
    ham_scores, spam_scores, predicted_labels, decision_labels = [], [], [], []

    for root, _, files in os.walk(TEST_PATH):
        for filename in files:
            actual_label = "ham" if "ham" in filename else "spam"
            full_path = os.path.join(root, filename)
            with open(full_path, encoding="latin-1") as file:
                email_words = text_parser(file.read())

                sigma_spam_score = sum(np.log(spam_cond_prob[word]) for word in email_words if word in all_unique_words)
                sigma_ham_score = sum(np.log(ham_cond_prob[word]) for word in email_words if word in all_unique_words)

                spam_score = np.log(spam_prob) + sigma_spam_score
                ham_score = np.log(ham_prob) + sigma_ham_score

                spam_scores.append(spam_score)
                ham_scores.append(ham_score)

                predicted_label = "spam" if spam_score > ham_score else "ham"
                predicted_labels.append(predicted_label)
                decision_labels.append("right" if predicted_label == actual_label else "wrong")

    return ham_scores, spam_scores, predicted_labels, decision_labels

def generate_result_output(file_count, file_names, predicted_labels, ham_scores, spam_scores, actual_labels, decision_labels):
    """Generates the content of result.txt."""
    return "\n".join(f"{i+1}  {file_names[i]}  {predicted_labels[i]}  {ham_scores[i]}  {spam_scores[i]}  {actual_labels[i]}  {decision_labels[i]}" 
                     for i in range(file_count))

def build_result_file(result_output):
    """Writes the result output to result.txt."""
    with open("result.txt", 'w') as result_file:
        result_file.write(result_output)

def calculate_precision(file_count, actual_labels, predicted_labels, label):
    """Calculates precision for the given label."""
    tp = sum(1 for i in range(file_count) if actual_labels[i] == label and predicted_labels[i] == label)
    fp = sum(1 for i in range(file_count) if actual_labels[i] != label and predicted_labels[i] == label)
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def calculate_recall(file_count, actual_labels, predicted_labels, label):
    """Calculates recall for the given label."""
    tp = sum(1 for i in range(file_count) if actual_labels[i] == label and predicted_labels[i] == label)
    fn = sum(1 for i in range(file_count) if actual_labels[i] == label and predicted_labels[i] != label)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calculate_accuracy(file_count, actual_labels, predicted_labels):
    """Calculates accuracy."""
    tp_tn = sum(1 for i in range(file_count) if actual_labels[i] == predicted_labels[i])
    return tp_tn / file_count

def calculate_f1_measure(precision, recall):
    """Calculates F1-measure."""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def get_confusion_params(file_count, actual_labels, predicted_labels, label):
    """Returns confusion parameters: (TP, TN, FP, FN)."""
    tp = sum(1 for i in range(file_count) if actual_labels[i] == label and predicted_labels[i] == label)
    tn = sum(1 for i in range(file_count) if actual_labels[i] != label and predicted_labels[i] != label)
    fp = sum(1 for i in range(file_count) if actual_labels[i] != label and predicted_labels[i] == label)
    fn = sum(1 for i in range(file_count) if actual_labels[i] == label and predicted_labels[i] != label)
    return tp, tn, fp, fn

def generate_evaluation_result(spam_accuracy, spam_precision, spam_recall, spam_f1, ham_accuracy, ham_precision, ham_recall, ham_f1):
    """Generates the evaluation results content."""
    return f"""
##################################################################################
#                           *** Evaluation Results ***                           #
#                                                                                #
#                  Accuracy |     Precision    | Recall |     F1-measure        #
# ==========================|===================|========|====================== #
#  Spam Class :    {spam_accuracy:.2f}   |{spam_precision:.2f} | {spam_recall:.2f}   | {spam_f1:.2f}    #
# --------------------------|-------------------|--------|---------------------- #
#  Ham Class  :    {ham_accuracy:.2f}   |{ham_precision:.2f} | {ham_recall:.2f}   | {ham_f1:.2f}    #
##################################################################################
"""

def build_evaluation_file(evaluation_output):
    """Writes the evaluation output to evaluation.txt."""
    with open("evaluation.txt", 'w') as evaluation_file:
        evaluation_file.write(evaluation_output)
