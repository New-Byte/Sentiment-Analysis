import categorize as cg

while True:
    try:
        print('Enter your comment: ')
        comment = input()
        closeres = ['close', 'exit', 'stop', '0','escape', 'esc']
        if(comment.lower() in closeres):
            break
        print('Comment you made is ' + cg.categorize(comment))
    except:
        print('Something went wrong!')
print('Analysis complete....')