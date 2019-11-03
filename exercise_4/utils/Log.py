def log(filename, content):
    with open('log/{}.log'.format(filename), 'a+') as file:
        file.write((str)(content))


def clearLog(filename):
    open('log/{}.log'.format(filename), 'w+').close()


def defaultLog(content):
    with open('log/log.log', 'a+') as file:
        file.write((str)(content) + '\n')

        
def clearDefaultLog():
    open('log/log.log', 'w+').close()