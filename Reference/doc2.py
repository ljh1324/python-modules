import glob

print(glob.glob.__doc__) # 이름 패턴을 써서 파일들을 찾는데에 사용된다
files = glob.glob('*py')
print(files)



