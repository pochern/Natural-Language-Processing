import re
import os.path


train_dir = input("Please enter the path to the \'train\' directory (do not leave a \'/\' at the end): ")
print("Thank you for the correct path to \'train\'! This program will now continue...")
pos_file_count = len(os.listdir(train_dir + "/pos"))
neg_file_count = len(os.listdir(train_dir + "/neg"))
total_file_count = pos_file_count + neg_file_count
pos_prior = pos_file_count/total_file_count
neg_prior = neg_file_count/total_file_count

pos_file_list = os.listdir(train_dir + "/pos")
neg_file_list = os.listdir(train_dir + "/neg")

if os.path.exists("mega-pos.txt"):
    os.remove("mega-pos.txt")
if os.path.exists("mega-neg.txt"):
    os.remove("mega-neg.txt")
if os.path.exists("mega-doc.txt"):
    os.remove("mega-doc.txt")


def form_mega_doc(file_list, mega_file_name, path):
    for filename in sorted(file_list):
        out_filename = mega_file_name
        with open(out_filename, 'a') as outfile:
            with open(path + '/' + filename, 'r') as infile:
                for line in infile:
                    line = re.sub(r'([.,!?"*]|((?:[:;=])(?:-o)?(?:[})DP8])|<br />|\s\())', r' \1 ', line)
                    line = re.sub(r'(\'\w|\w\w|\d\d)\)([.,\w\s])', r'\1 ) ', line)   # adds a space around closing parenthesis
                    line = re.sub(r'(\'\w|\w\w|\d\d)\((\w\'|\w\w|\d\d)', r'\1 ( ', line)   # adds a space around open parenthesis
                    line = re.sub(r'\s{2,}', ' ', line)   # reduces multiple spaces to 1
                    outfile.write(line.lower() + "\n")


form_mega_doc(pos_file_list, 'mega-pos.txt', train_dir + "/pos")
form_mega_doc(neg_file_list, 'mega-neg.txt', train_dir + "/neg")


def is_line_empty(line):
    return len(line.strip()) == 0

vocab_path = input("Please enter the path to the directory containing"
                   " the vocab file (do not leave a \'/\' at the end): ")
vocab_file = open(vocab_path + "/imdb.vocab", 'r')

vocab_vector = vocab_file.read().split()
vocab_len = len(vocab_vector)
vocab_file.close()

def get_feature_value(filename, out_filename):
    with open(out_filename, 'a') as outfile:
        with open(filename, 'r') as infile:
            for line in infile:
                line_split = line.split()
                feature_count = {}     # set all elements of list to 0
                for word in line_split:
                    if word in vocab_vector:
                        if word in feature_count:
                            feature_count[word] += 1
                        else:
                            feature_count[word] = 1
                if filename == "mega-pos.txt":
                    outfile.write("pos")
                if filename == "mega-neg.txt":
                    outfile.write("neg")
                for count in feature_count:
                    outfile.write(" " + count + ":" + str(feature_count[count]))
                outfile.write("\n")

get_feature_value("testing.txt", "mega-test.txt")


get_feature_value("mega-pos.txt", "mega-doc2.txt")
get_feature_value("mega-neg.txt", "mega-doc2.txt")

