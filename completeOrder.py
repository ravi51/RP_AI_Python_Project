class invalidcarError(Exception) :
    pass


def bill(ctype, price):
    menu ={"sport": 20 ,"hachback":10}
    try:
        if ctype not in menu:
            raise invalidcarError("this car is not present in not showroom")
        if not isinstance(price, int):
            raise TypeError("price must be integer ")
        
        total = menu[ctype] * price
        print(f"your order bill for {price} cups of {ctype} car :  rupees  {total}")

    except Exception as e:
        print("Error :", e)
    finally :
        print("Thank you for visiting again !")

bill("sport", 2)
bill("hachback", "three")
bill("c", 6)