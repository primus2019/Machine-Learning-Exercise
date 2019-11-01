def log(filename, content):
    with open('log/{}.log'.format(filename), 'a+') as file:
        file.write((str)(content))

def clearLog(filename):
    open('log/{}.log'.format(filename), 'w+').close()