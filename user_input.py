__all__ = ["get_float", "get_yes_no", "get_selection"]


def get_float(prompt: str) -> float:
    user_input = None

    while user_input is None:
        raw_selection = input(prompt)
        try:
            selection = float(raw_selection)
        except ValueError:
            print("Numbers only, please\r")
            selection = None

        if selection is not None:
            user_input = selection

    return user_input


def get_selection(selection_list: list) -> int:
    user_selection = None

    while user_selection is None:
        print("Selection:")
        for i, element in enumerate(selection_list):
            print(f"{i}:\t{element}")

        raw_selection = input()
        try:
            selection = int(raw_selection)
        except ValueError:
            print("Numbers only, please")
            selection = -1

        if selection in range(len(selection_list)):
            user_selection = selection

    return user_selection


def get_yes_no() -> bool:
    while True:
        raw_selection = input()
        if raw_selection.lower() in ["y", "yes"]:
            return True
        if raw_selection.lower() in ["n", "no"]:
            return False
