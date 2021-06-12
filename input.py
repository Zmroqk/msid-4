from typing import List


def get_positive_values(prompt="Provide values", use_float=False):
    values = []
    input_value = 0
    print(prompt)
    while len(values) == 0 or input_value != -1:
        try:
            if not use_float:
                input_value = int(input("Provide value (-1 ends): "))
            else:
                input_value = float(input("Provide value (-1 ends): "))
            if input_value > 0:
                values.append(input_value)
        except:
            pass
    return values

def ask_y_n(prompt="Choose y or n: "):
    input_value = ""
    while input_value != "y" and input_value != "n":
        input_value = input(prompt)
    return input_value == "y"


def menu(options: List[str], useExit: bool = True):
    elems = len(options)
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    if useExit:
        elems += 1
        print(f"{elems}. Exit")
    selected_option = 0
    while not 0 < selected_option < elems and not selected_option == -1:
        try:
            selected_option = int(input("Select option: "))
            if selected_option == elems:
                selected_option = -1
        except:
            pass
    return selected_option