class car:
    def __init__(self, seats):
        self._seats=seats

    @property # getter method
    def seats(self):
        return self._seats+1
    
    @seats.setter    # setter methods
    def seats(self, seats):
        if 1 <=seats <=7 :
            self._seats=seats
        else:
            raise ValueError("given seats car is not present")
        

car1 = car(4)
print(car1.seats)

car1.seats=7

print(car1.seats)