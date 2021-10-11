class Cat(object):
    def __init__(self, name='Kitty'):
        self.name = name

    def sayHi(self):
        print(self.name, 'says Hi!')

cat = Cat()

print(Cat.sayHi())