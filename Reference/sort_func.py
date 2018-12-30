# 정렬할 대상
fruits = ['apple', 'banana', 'kiwi', 'watermelon']

# 정렬을 위해서 과일 이름의 길이를 리턴하는 함수
def name_length(x):
    return len(x)

# 과일을 이름 순으로 정렬(함수 사용)
fruits.sort(key = name_length)
print(fruits)
# ['kiwi', 'apple', 'banana', 'watermelon']

# 람다를 사용한다면
# 람다는 다음과 같은 형식을 갖는다: lambda [매개변수]:[계산로직]
fruits.sort(key=lambda x: len(x))
print(fruits)
# ['kiwi', 'apple', 'banana', 'watermelon']