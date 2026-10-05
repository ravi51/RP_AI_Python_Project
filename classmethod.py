class car:
    def __init__(self, type_, seats):
        self.type= type_
        self.seats= seats
    
    @classmethod
    def honda(cls, order_dic):
        return cls(
            order_dic["type_"],
            order_dic["seats"],)
    
    @classmethod
    def sujuki(cls, order_str):
        type_, seats= order_str.split("-")
        return cls(type_, seats)

order1= car.honda({"type_":"sport" , "seats":"five"})
order2=car.sujuki("hachback-seven")

order3= car("tata", "six")
print(order1.__dict__)
print(order2.__dict__)
print(order3.__dict__)

