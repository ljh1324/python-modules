def function(*arglist):
    print(arglist)
    if not arglist:
        print('no arguments')
    else:
        for arg in arglist:
            if type(arg) == str:
                print("string!")
            elif type(arg) == list:
                print("list!")
            elif type(arg) == int:
                print("int!")
arr = [1, 2, 3]
function(1, 'a', arr)