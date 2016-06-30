def read(path):
    txt_file = open(path, 'r')
    txt = txt_file.read()
    txt_file.close()
    return txt