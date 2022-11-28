import numpy as np
#Poker Cash out

ALL_CARD_STR = np.array(['2H', '2D', '2C', '2S', '3H', '3D', '3C', '3S', '4H', '4D', '4C',
       '4S', '5H', '5D', '5C', '5S', '6H', '6D', '6C', '6S', '7H', '7D',
       '7C', '7S', '8H', '8D', '8C', '8S', '9H', '9D', '9C', '9S', 'TH',
       'TD', 'TC', 'TS', 'JH', 'JD', 'JC', 'JS', 'QH', 'QD', 'QC', 'QS',
       'KH', 'KD', 'KC', 'KS', 'AH', 'AD', 'AC', 'AS'])

NUMBER_PLAYER = 9
NUMBER_CARD = 52
NUMBER_BURN = 3
NUMBER_CARD_OPEN = 5
SMALL_CHIP = 1
BIG_CHIP = 2
ATTRIBUTE_PLAYER = 3
NUMBER_STATUS_GAME = 5        #(preflop, flop,, turn, river, showdown)


'''
env_state:
0-52: lá bài trong bộ bài, các lá bài còn nằm ở đầu, số là đã chia là số lá -1 nằm ở cuối
52-55: các lá bài burn, trống là -1
55-60: các lá open
60-69: tổng chip còn lại của các người chơi
69-78: chip đã bỏ ra trong lượt
78-87: tổng chip đã bỏ ra

87-96: trạng thái người chơi còn chơi hay k
96-105: lá bài thứ nhất của các người chơi
105-114: lá bài thứ hai của các người chơi
114-end:[button dealer, temp_button, status game, phase,
        id_action, cash to call_old, cash to call_new, sum pot, ván chơi thứ bao nhiêu

        ]
'''

INDEX = 0
#thẻ trên bàn
ENV_ALL_CARD_ON_BOARD= INDEX
INDEX += NUMBER_CARD

#card open
ENV_CARD_OPEN = INDEX
INDEX += NUMBER_CARD_OPEN

#chip of player
ENV_ALL_PLAYER_CHIP = INDEX
INDEX += NUMBER_PLAYER

#chip người chơi đã bỏ ra để theo
ENV_ALL_PLAYER_CHIP_GIVE = INDEX
INDEX += NUMBER_PLAYER

#tổng chip người chơi đã bỏ ra trong ván
ENV_ALL_PLAYER_CHIP_IN_POT = INDEX
INDEX += NUMBER_PLAYER

#player foled or still play
ENV_ALL_PLAYER_STATUS = INDEX
INDEX += NUMBER_PLAYER

#player first_card
ENV_ALL_FIRST_CARD = INDEX
INDEX += NUMBER_PLAYER

#player second_card
ENV_ALL_SECOND_CARD = INDEX
INDEX += NUMBER_PLAYER

#player first_card_showdown
ENV_ALL_FIRST_CARD_SHOWDOWN = INDEX
INDEX += NUMBER_PLAYER

#player second_card_showdown
ENV_ALL_SECOND_CARD_SHOWDOWN = INDEX
INDEX += NUMBER_PLAYER

#other in4
ENV_BUTTON_PLAYER = INDEX
INDEX += 1
ENV_TEMP_BUTTON = INDEX
INDEX += 1
ENV_STATUS_GAME = INDEX
INDEX += 1
ENV_PHASE = INDEX
INDEX += 1
ENV_ID_ACTION = INDEX
INDEX += 1
ENV_CASH_TO_CALL_OLD = INDEX
INDEX += 1
ENV_CASH_TO_CALL_NEW = INDEX
INDEX += 1
ENV_POT_VALUE = INDEX
INDEX += 1
ENV_NUMBER_GAME_PLAYED = INDEX
INDEX += 1
ENV_LENGTH = INDEX


'''
player_state:
0-52: lá bài trên bàn và của mình (giá trị 1 là của mình và đã mở, 0 là chưa mở và của người khác mình ko biết)
52-63: số chip còn lại của mỗi người
63:72: trạng thái người chơi (0 là đã bỏ game, 1 là còn chơi)
72:83: button (theo thứ tự từ phải qua trái của mình, tại đâu thì giá trị đó = 1)
83: cash to call
84: pot value
85: phase
86: status game
87: số ván đã chơi
'''

P_INDEX = 0
P_ALL_CARD = P_INDEX
P_INDEX += NUMBER_CARD*NUMBER_PLAYER

#chip of player
P_ALL_PLAYER_CHIP = P_INDEX
P_INDEX += NUMBER_PLAYER 

#chip người chơi đã bỏ ra để theo
P_ALL_PLAYER_CHIP_GIVE = P_INDEX
P_INDEX += NUMBER_PLAYER

#status of player
P_ALL_PLAYER_STATUS = P_INDEX
P_INDEX += NUMBER_PLAYER

#button
P_BUTTON_DEALER = P_INDEX
P_INDEX += NUMBER_PLAYER

#other in4
P_CASH_TO_CALL = P_INDEX
P_INDEX += 1

P_CASH_TO_BET = P_INDEX
P_INDEX += 1

P_POT_VALUE = P_INDEX
P_INDEX += 1

P_PHASE = P_INDEX
P_INDEX += 1

P_STATUS_GAME = P_INDEX
P_INDEX += NUMBER_STATUS_GAME

P_NUMBER_GAME_PLAY = P_INDEX
P_INDEX += 1

PLAYER_STATE_LENGTH = P_INDEX














