class A:
    def __init__(self):
        self.__x = 1

    def test(self):
        print(self.__x)


A().test()
print(A().__x)
