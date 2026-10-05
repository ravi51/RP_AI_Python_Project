"""file = open("order.txt","w")
try:
    file.write("sport car - 2 car")
finally:
    file.close()"""

with open("order1.txt" ,"w") as file:
    file.write("hackback car - 1 quantity")