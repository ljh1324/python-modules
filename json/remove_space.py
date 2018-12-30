
def remove_space(sentence):
    new_sentence = ""
    space_n = 0
    for c in sentence:
        if c == ' ':
            space_n += 1
        else:
            if space_n <= 1:
                new_sentence += c
                space_n = 0
            else:
                new_sentence += ' ' + c
                space_n = 0

    return new_sentence


def remove_space_in_txt(file, save_file):

    save_file = open(save_file, 'w')

    # add encoding='UTF-8' for this error "UnicodeDecodeError: 'cp949' codec can't decode byte 0xf6 in position 3767: illegal multibyte sequence"
    with open(file, 'rt', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            new_line = remove_space(line)
            save_file.write(new_line + '\n')

    save_file.close()

remove_space_in_txt('mood\\happy.txt', 'happy.txt')