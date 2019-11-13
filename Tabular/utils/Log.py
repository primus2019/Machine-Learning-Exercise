import os


def log(content, log_path='log/tmp.log', encoding='utf-8', delimeter='\n'):
    if not os.path.isdir('log/'):
        os.mkdir('log/')
    with open(log_path, 'a+', encoding=encoding) as file:
        file.write('{}{}'.format((str)(content), delimeter))


def clear_log(log_path='log/tmp.log'):
    if not os.path.isdir('log'):
        os.mkdir('log/')
    open(log_path, 'w+').close()
