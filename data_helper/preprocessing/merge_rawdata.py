import os

def merge_qa_lists(dir_path, output_file_path):

    '''
    to merge all the qa files in given dir and output them as one file.
    
    :param dir_path: 
    :return: 
    '''

    qa_pairs = []

    for filename in os.listdir(dir_path):
        read_qa_pairs(os.path.join(dir_path, filename), qa_pairs)

    ctr = 0

    with open(output_file_path, 'w', encoding='utf-8') as output:
        for q,a in qa_pairs:
            output.write(q + '\t' + a + '\n')
            ctr += 1
            if ctr % 1000 == 0:
                print("Has outputed %d results." % (ctr))

def read_qa_pairs(path, qa_pairs):

    print("Going to read the qa-list", path)

    with open(path, 'r', encoding='utf-8') as qa_list:
        for line in qa_list:
            line = line.strip('\n')
            qa_pairs.append(line.split('\t'))

    print("The processing of", path, "is over. len(qa_pairs) =", len(qa_pairs))


def main():

    merge_qa_lists('../corpus/', output_file_path='../data/Gossiping-QA-Data.txt')

if __name__ == "__main__":
    main()