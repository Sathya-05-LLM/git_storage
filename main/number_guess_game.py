def number_guess_game():
    print("Welcome to the Number Guessing Game!")
    print("I'm thinking of a number between 1 and 100.")
    import random

    secret_numbner = random.randint(1, 100)

    for i in range(0,7):
        try:
            user_input = int(input("Make a guess: "))
            if user_input < i or user_input > 100:
                print("Please enter a number between 1 and 100.")
                continue
            elif user_input < secret_numbner:
                print("Too low.Try again")
            elif user_input > secret_numbner:
                print("Too high.Try again")
            else:
                print(f"You got it! The answer was {secret_numbner}.")
                break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    if secret_numbner == user_input:
        print(f"You got it! The answer was {secret_numbner}.")
    else:
        print("You've used all your attempts. Try again next time.")

if __name__ == "__main__":
    number_guess_game()