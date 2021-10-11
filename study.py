class Cat(object):
    def __init__(self, name):
        self.name = name

    def sayHi(self):
        print(self.name, 'says Hi!')
    
    def change(self,st):
        self.name = st

def cc(oj):
    oj.change('jack')

cat = Cat("tom")
cat.sayHi()
cc(cat)
cat.sayHi()

