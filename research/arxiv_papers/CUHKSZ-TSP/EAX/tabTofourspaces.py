with open('syngen_new.py', 'r') as file:
    content = file.read()

content = content.replace('\t', ' ' * 4)

with open('syngen_new.py', 'w') as file:
    file.write(content)