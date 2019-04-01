import pre_process as pp
import re
# pp.form_mega_doc()
# training_file = "dkfj"
# test_file = "dfg"
# model_param_file = "dfg"
# output_file = "dfg"

# 1. Movie review classification using Naive Bayes
# Sentence: "I always like foreign films"
# Cnb=P(cj)*PRODUCTofP(x|c)
# P(pos-prior)*P(I|pos)*P(always|pos)*P(like|pos)*P(foreign|pos)*P(films|pos)
pos_nb = 0.4*0.09*0.07*0.29*0.04*0.08
# P(neg-prior)*P(I|neg)*P(always|neg)*P(like|neg)*P(foreign|neg)*P(films|neg)
neg_nb = 0.6*0.16*0.06*0.06*0.15*0.11
print("Answer to Question 1:")
if pos_nb > neg_nb:
    print("Naive Bayes will assign class pos to the sentence \"I always like foreign films\".\n")
else:
    print("Naive Bayes will assign class neg to the sentence \"I always like foreign films\".\n")


training_file = open("training.txt", 'r')
comedy = 0
action = 0
comedy_total_count = 0
action_total_count = 0
comedy_dict = {}
action_dict = {}

for line in training_file:
    line = re.sub(r':+(\d)', r' \1', line)
    line_split = line.split()
    if line_split[0] == "comedy":
        comedy += 1
        for i in range(1, len(line_split), 2):
            key = line_split[i]
            value = line_split[i + 1]
            comedy_total_count += int(value)
            if key in comedy_dict:
                comedy_dict[key] += int(value)
            else:
                comedy_dict[key] = int(value)
    if line_split[0] == "action":
        action += 1
        for i in range(1, len(line_split), 2):
            key = line_split[i]
            value = line_split[i + 1]
            action_total_count += int(value)
            if key in action_dict:
                action_dict[key] += int(value)
            else:
                action_dict[key] = int(value)

# print(comedy_total_count)
# print(action_total_count)
total = comedy + action
prior_comedy = comedy / total
prior_action = action / total
# print(str(prior_comedy))
# print(str(prior_action))
vocab_list = ["fun", "couple", "love", "fast", "furious", "shoot", "fly"]
# print(len(vocab_list))
test_file = open("test.txt", 'r')
c_nb = 0
p_comedy = 0.0
p_action = 0.0


def nb_classifier(label_prob, label_dict, line_var, label_total_count, vocab_var, label_prior):
    if line_var[0] in label_dict:
        label_prob += (label_dict[line_var[0]] + 1) / (label_total_count + len(vocab_var))
        # print("label_prob: " + str(label_prob))
    else:
        label_prob += (0 + 1) / (label_total_count + len(vocab_var))
    for i in range(1, len(line_var), 1):
        if line_var[i] in label_dict:
            label_prob *= (label_dict[line_var[i]] + 1) / (label_total_count + len(vocab_var))
            # print("label_prob: " + str(label_prob))
        else:
            label_prob *= (0 + 1) / (label_total_count + len(vocab_var))
            # print("no label_prob: " + str(label_prob))
    c_nb = label_prob * label_prior
    return c_nb


for line in test_file:
    line_split = line.split()
    comedy_nb = nb_classifier(p_comedy, comedy_dict, line_split, comedy_total_count, vocab_list, prior_comedy)
    action_nb = nb_classifier(p_action, action_dict, line_split, action_total_count, vocab_list, prior_action)
    print("Answer to Question 2(c):")
    print("The probability for class \'comedy\' is " + str(comedy_nb))
    print("The probability for class \'action\' is " + str(action_nb))
    if (comedy_nb > action_nb):
        print("The most likely class for \"" + line + "\" is \'comedy\'")
    else:
        print("The most likely class for \"" + line + "\" is \'action\'")

test_file.close()

pos_total_count = 0
neg_total_count = 0
pos = 0
neg = 0
pos_dict = {}
neg_dict = {}
training_file = open("mega-doc2.txt", 'r')
# 2d. Train classifier on the training data
for line in training_file:
    line = re.sub(r':+(\d)', r' \1', line)
    line_split = line.split()
    if line_split[0] == "pos":
        pos += 1
        for i in range(1, len(line_split), 2):
            key = line_split[i]
            value = line_split[i + 1]
            pos_total_count += int(value)
            if key in pos_dict:
                pos_dict[key] += int(value)
            else:
                pos_dict[key] = int(value)
    if line_split[0] == "neg":
        neg += 1
        for i in range(1, len(line_split), 2):
            key = line_split[i]
            value = line_split[i + 1]
            neg_total_count += int(value)
            if key in neg_dict:
                neg_dict[key] += int(value)
            else:
                neg_dict[key] = int(value)