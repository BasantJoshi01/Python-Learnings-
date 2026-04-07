import random 


def get_choise():
    player_choise = input("Enter a Choise ( Rock , Paper , scissor ) : ")

    Choicees = ["Rock" , "Paper" , "Scissors"]
    computer_choise = random.choices(Choicees)
    choices  = { "Player" : player_choise , "Computer" : computer_choise}
    return choices

def CheckWin( Player , Computer ):
        
        # print( "You Chose"  +  Player  +  "Computer gave" + Computer )
        if Player == Computer:
                return " It's a Tie! "
        else :
             return[Player , Computer]

result = get_choise()
print(CheckWin( result["Player"] , result["Computer"] ))


