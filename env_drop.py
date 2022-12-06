import numba as nb
import numpy as np
from index import*

@nb.njit()
def initEnv():
    env_state = np.zeros(ENV_LENGTH)
    env_state[ENV_CARD_OPEN : ENV_ALL_PLAYER_CHIP] = -1
    env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE] = np.full(NUMBER_PLAYER, 100*BIG_CHIP)
    env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD] = np.ones(NUMBER_PLAYER)
    env_state[ENV_ALL_FIRST_CARD : ENV_BUTTON_PLAYER] = np.full(4*NUMBER_PLAYER, -1)
    env_state[ENV_BUTTON_PLAYER] = -1            #người chơi ở vị trí đầu tiên giữ button
    return env_state

@nb.njit()
def reset_round(old_env_state):
    env_state = np.zeros(ENV_LENGTH)
    #tính toán chip còn lại của người chơi, từ đó tính ra trạng thái của người chơi ở game mới
    env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE] = old_env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE]
    env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD] = (env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE] > 0) * 1

    #thiết lập button dealer và temp button, trừ chip của small và big để cộng vào sum_pot_value, thiết lập id_action, status_game, phase, số ván đã chơi
    env_state[ENV_BUTTON_PLAYER] = (old_env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
    while env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_BUTTON_PLAYER])] == 0:
        env_state[ENV_BUTTON_PLAYER] = (env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
    sum_pot = 0
    #trừ chip của small_player
    sm_player = (env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
    while env_state[int(ENV_ALL_PLAYER_CHIP + sm_player)] == 0:
        sm_player = (sm_player + 1) % NUMBER_PLAYER
    sum_pot += SMALL_CHIP
    env_state[int(ENV_ALL_PLAYER_CHIP + sm_player)] -= SMALL_CHIP
    env_state[int(ENV_ALL_PLAYER_CHIP_GIVE + sm_player)] += SMALL_CHIP
    env_state[int(ENV_ALL_PLAYER_CHIP_IN_POT + sm_player)] += SMALL_CHIP
    #print('update chip give')
    
    #trừ chip của big_player
    big_player = (sm_player + 1) % NUMBER_PLAYER
    while env_state[int(ENV_ALL_PLAYER_CHIP + big_player)] == 0:
        big_player = (big_player + 1) % NUMBER_PLAYER
    sum_pot += min(BIG_CHIP, env_state[int(ENV_ALL_PLAYER_CHIP + big_player)])
    env_state[int(ENV_ALL_PLAYER_CHIP + big_player)] -= min(2, env_state[int(ENV_ALL_PLAYER_CHIP + big_player)])
    env_state[int(ENV_ALL_PLAYER_CHIP_GIVE + big_player)] += BIG_CHIP
    env_state[int(ENV_ALL_PLAYER_CHIP_IN_POT + big_player)] += BIG_CHIP

    temp_button = (big_player + 1) % NUMBER_PLAYER
    while env_state[int(ENV_ALL_PLAYER_CHIP + temp_button)] == 0:
        temp_button = (temp_button + 1) % NUMBER_PLAYER
    env_state[ENV_TEMP_BUTTON] = temp_button
    

    env_state[ENV_CASH_TO_CALL_NEW] = BIG_CHIP
    env_state[ENV_CASH_TO_CALL_OLD] = BIG_CHIP
    env_state[ENV_POT_VALUE] = sum_pot

    #người chơi action kế tiếp là người thứ 1 trở đi từ button dealer mà còn chip
    env_state[ENV_ID_ACTION] = temp_button
    # while count < 3 and env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])] > 0:
    #     env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER
    #     if env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])] > 0:
    #         count += 1
    # print('check env', env_state[ENV_ID_ACTION], count)

    while env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])] == 0:
        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER
    # env_state[ENV_PHASE] = 0
    # env_state[ENV_STATUS_GAME] = 0
    env_state[ENV_NUMBER_GAME_PLAYED] = old_env_state[ENV_NUMBER_GAME_PLAYED] + 1
    #reset 2 lá bài của người chơi và bài showdown, sau đó chia bài
    env_state[ENV_ALL_FIRST_CARD : ENV_BUTTON_PLAYER] = np.full(4*NUMBER_PLAYER, -1)
    all_card_num = np.arange(52)
    np.random.shuffle(all_card_num)
    env_state[ENV_CARD_OPEN : ENV_ALL_PLAYER_CHIP] = all_card_num[:NUMBER_CARD_OPEN]
    env_state[ENV_ALL_FIRST_CARD : ENV_ALL_FIRST_CARD_SHOWDOWN] = all_card_num[NUMBER_CARD_OPEN : NUMBER_CARD_OPEN + 2 * NUMBER_PLAYER]
    env_state[ENV_ALL_CARD_ON_BOARD : ENV_CARD_OPEN] = np.append(all_card_num[NUMBER_CARD_OPEN + 2 * NUMBER_PLAYER : ], np.full(NUMBER_CARD_OPEN + 2 * NUMBER_PLAYER, -1))
    return env_state

@nb.njit()
def getAgentState(env_state):
    player_state = np.zeros(PLAYER_STATE_LENGTH)
    id_action = int(env_state[ENV_ID_ACTION])
    #cập nhật chip còn lại
    all_player_chip = env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE]
    player_state[P_ALL_PLAYER_CHIP : P_ALL_PLAYER_CHIP_GIVE] = np.concatenate((all_player_chip[id_action: ], all_player_chip[:id_action]))
    #Cập nhật chip đã bỏ ra
    all_player_chip_give = env_state[ENV_ALL_PLAYER_CHIP_GIVE : ENV_ALL_PLAYER_CHIP_IN_POT]
    player_state[P_ALL_PLAYER_CHIP_GIVE : P_ALL_PLAYER_STATUS] = np.concatenate((all_player_chip_give[id_action: ], all_player_chip_give[:id_action]))
    #cập nhật status
    all_player_status = env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD]
    player_state[P_ALL_PLAYER_STATUS : P_BUTTON_DEALER] = np.concatenate((all_player_status[id_action: ], all_player_status[:id_action]))
    #cập nhật button dealer
    player_state[P_BUTTON_DEALER + int(env_state[ENV_BUTTON_PLAYER] - id_action) % NUMBER_PLAYER] = 1
    #cập nhật chip to call
    player_state[P_CASH_TO_CALL] = max(0,env_state[ENV_CASH_TO_CALL_OLD] - env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action])
    #cập nhật chip to bet
    player_state[P_CASH_TO_BET] = env_state[ENV_CASH_TO_CALL_OLD] 
    #cập nhật pot value, phase, status game
    player_state[P_POT_VALUE] = env_state[ENV_POT_VALUE]
    # player_state[int(P_PHASE + env_state[ENV_PHASE])] = 1
    player_state[P_PHASE] = env_state[ENV_PHASE]
    player_state[int(P_STATUS_GAME + max(env_state[ENV_STATUS_GAME] - 2, 0))] = 1
    player_state[P_NUMBER_GAME_PLAY] = env_state[ENV_NUMBER_GAME_PLAYED]

    #cập nhật card
    player_card = np.array([env_state[ENV_ALL_FIRST_CARD + id_action], env_state[ENV_ALL_SECOND_CARD + id_action]]).astype(np.int64)
    if env_state[ENV_STATUS_GAME] == 0:
        player_state[P_ALL_CARD : P_ALL_CARD + NUMBER_CARD][player_card] = 1 
    elif env_state[ENV_STATUS_GAME] != 6:
        #nếu ko phải showdown và pre flop
        card_open = env_state[ENV_CARD_OPEN : int(ENV_CARD_OPEN + env_state[ENV_STATUS_GAME])].astype(np.int64)
        for id in range(1,NUMBER_PLAYER):
            status_id = player_state[P_ALL_PLAYER_STATUS : P_BUTTON_DEALER][id]
            if status_id == 1:
                player_state[P_ALL_CARD + NUMBER_CARD * id : P_ALL_CARD + NUMBER_CARD * (id + 1)][card_open] = 1 

        player_card = np.append(player_card, card_open)
        player_state[P_ALL_CARD : P_ALL_CARD + NUMBER_CARD][player_card] = 1 

    else:
        #duyệt từng người, ông nào tham gia showdown thì cho xem bài, ông nào bài xấu ko mở thì đã bị gán thành -1 ở step, nên lọc bài lớn hơn -1 là đc
        card_open = env_state[ENV_CARD_OPEN : ENV_ALL_PLAYER_CHIP].astype(np.int64)
        for id in range(NUMBER_PLAYER):
            status_id = player_state[P_ALL_PLAYER_STATUS : P_BUTTON_DEALER][id]
            if status_id == 1:
                id_env = (id_action - id) % NUMBER_PLAYER
                player_i_card = np.array([env_state[ENV_ALL_FIRST_CARD_SHOWDOWN + id_env], env_state[ENV_ALL_SECOND_CARD_SHOWDOWN + id_env]]).astype(np.int64)
                player_i_card = np.append(player_i_card, card_open)
                player_i_card = player_i_card[player_i_card > -1]
                player_state[P_ALL_CARD + NUMBER_CARD * id : P_ALL_CARD + NUMBER_CARD * (id + 1)][player_i_card] = 1 

    return player_state

@nb.njit()
def getValidActions(player_state):
    list_action = np.zeros(6)
    if player_state[P_PHASE] == 0:
        #nếu cash to call == 0 thì check, bet, allin
        if player_state[P_CASH_TO_CALL] == 0:
            list_action[1:5] = 1
            list_action[2] = 0
        #nếu cash to call >= 0
        else:
            #nếu cash_to_call > cash (fold, allin)
            if player_state[P_ALL_PLAYER_CHIP] <= player_state[P_CASH_TO_CALL]:
                #fold, allin
                list_action[2] = 1
                list_action[4] = 1
            elif player_state[P_ALL_PLAYER_CHIP] > player_state[P_CASH_TO_CALL] and player_state[P_ALL_PLAYER_CHIP] - player_state[P_CASH_TO_CALL] < player_state[P_CASH_TO_BET]:
                #fold, call, allin
                list_action[0] = 1
                list_action[2] = 1
                list_action[4] = 1
            elif player_state[P_ALL_PLAYER_CHIP] -  player_state[P_CASH_TO_CALL] >= player_state[P_CASH_TO_BET]:
                #call, fold, bet, allin
                list_action[:5] = 1
                list_action[1] = 0
            else:
                raise Exception('xét thieu trường hợp')
    #nếu đang bet dở thì bet, allin, dừng bet
    elif player_state[P_PHASE] == 1:
        if player_state[P_ALL_PLAYER_CHIP] >= player_state[P_CASH_TO_BET]:
            list_action[3 :] = 1
        else:
            list_action[4 :] = 1
    return list_action

@nb.njit()
def stepEnv(env_state, action):
    phase_env = int(env_state[ENV_PHASE])
    id_action = int(env_state[ENV_ID_ACTION])
    actionstrlst = ['call', 'check', 'fold', 'betraise', 'allin', 'stopbet']
    # print(id_action, 'phase', phase_env, action_str, 'button', env_state[ENV_TEMP_BUTTON], env_state[ENV_BUTTON_PLAYER], 'status', int(env_state[ENV_STATUS_GAME]), env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE], env_state[ENV_ALL_PLAYER_CHIP_IN_POT : ENV_ALL_PLAYER_STATUS], env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD])
    # #print(id_action, phase_env, action, 'status', int(env_state[ENV_STATUS_GAME]), env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE], env_state[ENV_TEMP_BUTTON])
    if phase_env == 0:
        if action == 0:     #người chơi call
            #trừ chip người chơi và tăng giá trị pot, bổ sung giá trị số tiền đã bỏ ra
            chip_to_call = env_state[ENV_CASH_TO_CALL_OLD] - env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action]
            env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action] += chip_to_call
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT+id_action] += chip_to_call
            env_state[ENV_ALL_PLAYER_CHIP + id_action] -= chip_to_call
            env_state[ENV_POT_VALUE] += chip_to_call
        elif action == 1:   #người chơi check
            pass
        elif action == 2:   #người chơi fold
            #nếu người chơi fold, chuyển trạng thái của người đó.
            env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action] = 0
            env_state[ENV_ALL_PLAYER_STATUS + id_action] = 0
        elif action == 3:   #người chơi bet/raise
            #nếu người chơi bet, trừ tiền người chơi, tăng tiền pot, tăng cash to call_new
            #bổ sung giá trị số tiền đã bỏ ra
            chip_to_bet_raise = 0
            if env_state[ENV_CASH_TO_CALL_OLD] == 0:    #nếu chưa ai bet thì mình bet 
                env_state[ENV_CASH_TO_CALL_OLD] = SMALL_CHIP
                chip_to_bet_raise = SMALL_CHIP
            else:           #nếu đã có người bet thì mình raise
                chip_to_bet_raise = 2*env_state[ENV_CASH_TO_CALL_OLD] - env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action]

            env_state[ENV_ALL_PLAYER_CHIP + id_action] -= chip_to_bet_raise
            env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action] += chip_to_bet_raise
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT+id_action] += chip_to_bet_raise
            env_state[ENV_POT_VALUE] += chip_to_bet_raise
            env_state[ENV_CASH_TO_CALL_NEW] += env_state[ENV_CASH_TO_CALL_OLD]
            #nếu bet thì sang phase xem có bet tiếp k

            env_state[ENV_TEMP_BUTTON] = id_action
            #print('gán lại temp button', id_action, env_state[ENV_TEMP_BUTTON])
            env_state[ENV_PHASE] = 1
        elif action == 4:
            #nếu người chơi all_in, bổ sung giá trị chip người chơi bỏ ra và tăng pot
            chip_to_allin = env_state[ENV_ALL_PLAYER_CHIP + id_action]
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT+id_action] += chip_to_allin
            env_state[ENV_POT_VALUE] += chip_to_allin
            #điều chỉnh cash to call nếu cần, điều chỉnh button nếu người chơi bet nhiều chip nhất
            if env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action] + chip_to_allin > np.max(env_state[ENV_ALL_PLAYER_CHIP_GIVE : ENV_ALL_PLAYER_CHIP_IN_POT]):
                env_state[ENV_CASH_TO_CALL_OLD] = chip_to_allin
                env_state[ENV_CASH_TO_CALL_NEW] = env_state[ENV_CASH_TO_CALL_OLD]
                env_state[ENV_TEMP_BUTTON] = id_action
            env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action] += chip_to_allin
            #trừ hết tiền người chơi all in
            env_state[ENV_ALL_PLAYER_CHIP + id_action] = 0

        if action != 3:
            id_action_next = (id_action + 1) % NUMBER_PLAYER
            '''
            hết lượt của người chơi hiện tại, chuyển người chơi, loop đến khi tìm đc người kế tiếp có thể action loop trở về đến
            vị trí hiện tại thì cũng dừng
            '''
            if id_action_next != env_state[ENV_TEMP_BUTTON]:
                while env_state[ENV_ALL_PLAYER_STATUS + id_action_next] == 0 or env_state[ENV_ALL_PLAYER_CHIP + id_action_next] == 0:
                    id_action_next = (id_action_next + 1) % NUMBER_PLAYER
                    #print('while 1')
                    if id_action_next == env_state[ENV_TEMP_BUTTON]:
                        break
            #print('check', id_action_next, env_state[ENV_STATUS_GAME], env_state[ENV_TEMP_BUTTON])
            if id_action_next == env_state[ENV_TEMP_BUTTON]:
                #kết thúc vòng chơi, chuyển status game, reset give chip, xác định id_action mới là người còn chơi và còn chip. Nếu hết 1 vòng => các người chơi đã allin hết, đi thẳng đến showdown
                if env_state[ENV_STATUS_GAME] != 5:
                    env_state[ENV_ALL_PLAYER_CHIP_GIVE : ENV_ALL_PLAYER_CHIP_IN_POT] = 0
                    env_state[ENV_ID_ACTION] = (env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
                    env_state[ENV_CASH_TO_CALL_OLD], env_state[ENV_CASH_TO_CALL_NEW] = 0, 0
                    #loop cho đến khi gặp người vừa còn chơi và vừa còn chip
                    while env_state[int(ENV_ALL_PLAYER_STATUS + env_state[ENV_ID_ACTION])] == 0 or env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])] == 0:
                        #print('while 2', env_state[ENV_ID_ACTION], env_state[ENV_BUTTON_PLAYER], env_state[int(ENV_ALL_PLAYER_STATUS + env_state[ENV_ID_ACTION])], env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])], env_state[ENV_BUTTON_PLAYER] )
                        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER
                        #nếu mọi người hết khả năng action tiếp thì đi thẳng đến showdown
                        if env_state[ENV_ID_ACTION] == env_state[ENV_BUTTON_PLAYER]:
                            # if np.count_nonzero(env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD]) != 1:
                            env_state[ENV_STATUS_GAME] = 6
                            #print('gnas đi showdown')
                            env_state[ENV_ID_ACTION] = env_state[ENV_TEMP_BUTTON]
                            break
                    if env_state[ENV_STATUS_GAME] != 6:
                        temp_button = (env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
                        while env_state[int(ENV_ALL_PLAYER_STATUS + temp_button)] == 0:
                            temp_button = (temp_button + 1) % NUMBER_PLAYER
                        env_state[ENV_TEMP_BUTTON] = temp_button

                if env_state[ENV_STATUS_GAME] == 0:
                    env_state[ENV_STATUS_GAME] = 3
                elif env_state[ENV_STATUS_GAME] == 3:
                    env_state[ENV_STATUS_GAME] = 4
                elif env_state[ENV_STATUS_GAME] == 4:
                    env_state[ENV_STATUS_GAME] = 5
                elif env_state[ENV_STATUS_GAME] == 5:
                    env_state[ENV_STATUS_GAME] = 6

                #print('test', env_state[ENV_STATUS_GAME])
                if env_state[ENV_STATUS_GAME] == 6:
                    env_state = showdown(env_state)
                    env_state[ENV_ID_ACTION] = env_state[ENV_TEMP_BUTTON]
                    #print('showdown')
            else:
                #cập nhật người chơi
                env_state[ENV_ID_ACTION] = id_action_next
                

    elif phase_env == 1:
        if action == 3:
            #nếu người chơi bet tiếp, trừ tiền người chơi, tăng tiền pot, tăng cash to call_new
            #bổ sung giá trị số tiền đã bỏ ra
            chip_to_bet_raise = env_state[ENV_CASH_TO_CALL_OLD]
            env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action] += chip_to_bet_raise
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT+id_action] += chip_to_bet_raise
            env_state[ENV_POT_VALUE] += chip_to_bet_raise
            env_state[ENV_CASH_TO_CALL_NEW] += chip_to_bet_raise
            env_state[ENV_ALL_PLAYER_CHIP + id_action] -= chip_to_bet_raise
        elif action == 4:
            #nếu all in khi bet tiếp, update chip give, pot value, cash_to_call_old and cash_to_call_new
            chip_to_allin = env_state[ENV_ALL_PLAYER_CHIP + id_action]
            env_state[ENV_ALL_PLAYER_CHIP + id_action] = 0
            env_state[ENV_ALL_PLAYER_CHIP_GIVE+id_action] += chip_to_allin
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT+id_action] += chip_to_allin
            env_state[ENV_POT_VALUE] += chip_to_allin
            env_state[ENV_CASH_TO_CALL_OLD] = env_state[ENV_CASH_TO_CALL_NEW] + chip_to_allin
            env_state[ENV_CASH_TO_CALL_NEW] = env_state[ENV_CASH_TO_CALL_OLD]
            env_state[ENV_TEMP_BUTTON] = id_action
        elif action == 5:
            env_state[ENV_CASH_TO_CALL_OLD] = env_state[ENV_CASH_TO_CALL_NEW] 
            env_state[ENV_CASH_TO_CALL_NEW] = env_state[ENV_CASH_TO_CALL_OLD]
            env_state[ENV_TEMP_BUTTON] = id_action

        if action != 3:
            '''
            hết lượt của người chơi hiện tại, chuyển người chơi, loop đến khi tìm đc người kế tiếp có thể action loop trở về đến
            vị trí hiện tại thì cũng dừng
            '''
            id_action_next = (id_action + 1) % NUMBER_PLAYER
            while env_state[ENV_ALL_PLAYER_STATUS + id_action_next] == 0 or env_state[ENV_ALL_PLAYER_CHIP + id_action_next] == 0:
                id_action_next = (id_action_next + 1) % NUMBER_PLAYER
                #print('while 3')
                if id_action_next == env_state[ENV_TEMP_BUTTON]:
                    break
            if id_action_next == env_state[ENV_TEMP_BUTTON]:
                #kết thúc vòng chơi, chuyển status game, reset give chip, xác định id_action mới là người còn chơi và còn chip. Nếu hết 1 vòng => các người chơi đã allin hết, đi thẳng đến showdown
                if env_state[ENV_STATUS_GAME] != 5:
                    env_state[ENV_ALL_PLAYER_CHIP_GIVE : ENV_ALL_PLAYER_CHIP_IN_POT] = 0
                    env_state[ENV_ID_ACTION] = (env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
                    env_state[ENV_CASH_TO_CALL_OLD], env_state[ENV_CASH_TO_CALL_NEW] = 0, 0
                    #loop cho đến khi gặp người vừa còn chơi và vừa còn chip
                    while env_state[int(ENV_ALL_PLAYER_STATUS + env_state[ENV_ID_ACTION])] == 0 or env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])] == 0:
                        #print('while 2', env_state[ENV_ID_ACTION], env_state[ENV_BUTTON_PLAYER], env_state[int(ENV_ALL_PLAYER_STATUS + env_state[ENV_ID_ACTION])], env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])], env_state[ENV_BUTTON_PLAYER] )
                        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER
                        #nếu mọi người hết khả năng action tiếp thì đi thẳng đến showdown
                        if env_state[ENV_ID_ACTION] == env_state[ENV_BUTTON_PLAYER]:
                            # if np.count_nonzero(env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD]) != 1:
                            env_state[ENV_STATUS_GAME] = 6
                            #print('gnas đi showdown')
                            env_state[ENV_ID_ACTION] = env_state[ENV_TEMP_BUTTON]
                            break
                    if env_state[ENV_STATUS_GAME] != 6:
                        temp_button = (env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
                        while env_state[int(ENV_ALL_PLAYER_STATUS + temp_button)] == 0:
                            temp_button = (temp_button + 1) % NUMBER_PLAYER
                        env_state[ENV_TEMP_BUTTON] = temp_button
             

                if env_state[ENV_STATUS_GAME] == 0:
                    env_state[ENV_STATUS_GAME] = 3
                elif env_state[ENV_STATUS_GAME] == 3:
                    env_state[ENV_STATUS_GAME] = 4
                elif env_state[ENV_STATUS_GAME] == 4:
                    env_state[ENV_STATUS_GAME] = 5
                elif env_state[ENV_STATUS_GAME] == 5:
                    env_state[ENV_STATUS_GAME] = 6


                if env_state[ENV_STATUS_GAME] == 6:
                    env_state = showdown(env_state)
                    env_state[ENV_ID_ACTION] = env_state[ENV_TEMP_BUTTON]
                    
            else:
                #cập nhật người chơi
                env_state[ENV_ID_ACTION] = id_action_next
                env_state[ENV_PHASE] = 0
    
    return env_state

@nb.njit()
def combinations_using_numba(pool, r):
    n = len(pool)
    indices = list(range(r))
    empty = not(n and (0 < r <= n))
    if not empty:
        result = [pool[i] for i in indices]
        yield result
    while not empty:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            result = [pool[i] for i in indices]
            yield result

@nb.njit()
def evaluate_num_numba(hand, id_player):
    '''
    input:  - hand: các lá bài của người chơi 
            - id_player: index của người chơi trong danh sách người chơi
    out_put: list gồm:  + điểm của hand bài
                        + rank kicker
                        + danh sách bài trong hand
                        +id người chơi
    '''
    hand = hand.astype(np.int64)
    # ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    rank_card = hand // 4
    suit_card = hand % 4
    all_index_card = [0,1,2,3,4,5,6]
    all_score = []
    all_hands = np.array(list(combinations_using_numba(all_index_card, 5)))
    for id in range(len(all_hands)):
        sm_hand = all_hands[id]
        rank_sm_hand = rank_card[sm_hand]
        suit_sm_hand = suit_card[sm_hand]
        score = np.array([0, 0, 0, 0, 0])
        rankss = [-1, -1, -1, -1, -1]
        arr_rank_score = [[0,-1], [0,-1], [0,-1], [0,-1], [0,-1]]
        for i in range(5):
            rank = rank_sm_hand[i]
            if rank not in rankss:
                rankss[i] = rank
                arr_rank_score[i] = [len(rank_sm_hand[rank_sm_hand == rank]), rank]
        arr_rank_score = np.array(sorted(arr_rank_score, reverse=True))
        rankss = list(arr_rank_score[:,1])
        score = arr_rank_score[:,0]
        if len(score[score > 0]) == 5: # if there are 5 different ranks it could be a straight or a flush (or both)
            if rankss[0:2] == [12, 3]: 
                rankss = [3, 2, 1, 0, -1] # adjust if 5 high straight
            all_type_hand = [[np.array([1,0,0,0,0]),np.array([3,1,2,0,0])],[np.array([3,1,3,0,0]),np.array([5,0,0,0,0])]]
            score = all_type_hand[int(len(np.unique(suit_sm_hand)) == 1)][int(rankss[0] - rankss[4] == 4)] # high card, straight, flush, straight flush
        score = list(score)
        sm_hand = list(hand[sm_hand])
        all_score.append([score, rankss, sm_hand, [id_player,-1,-1,-1,-1]])
    return max(all_score)

@nb.njit()
def holdem(board, hands):
    '''
    board: các thẻ bài chung
    hands: list các bộ bài của từng người chơi
    '''
    # scores = []
    scores = [[[-1]*5]*4 for i in range(9)]
    for i in range(len(hands)):
        result = evaluate_num_numba(np.append(board, hands[i]), i)
        # scores.append(result)
        scores[i] = result

    all_best_hand_player =np.zeros((NUMBER_PLAYER,5))

    for i in range(len(scores)):
        all_best_hand_player[i] = np.array(scores[i][2])
    # print(all_best_hand_player)
    scores = sorted(scores, reverse= True)

    # best, hand, id = max(scores)[0],  max(scores)[2], max(scores)[3]
    top = np.zeros((9,9))
    topth = 0
    top[topth][scores[0][3][0]] = 1
    for i in range(len(scores) - 1):
        if scores[i][0] == scores[i+1][0]:
            if scores[i][1] == scores[i+1][1]:
                top[topth][scores[i+1][3][0]] = 1
            else:
                topth += 1
                top[topth][scores[i+1][3][0]] = 1
        else:
            topth += 1
            top[topth][scores[i+1][3][0]] = 1
    # print(sys.getsizeof(scores))
    # print('best', best, ALL_CARD_STR[np.array(hand)], hand, 'ID', id)
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # print(top)
    # scores = scores[:1]
    ranks = np.full(9,-1)
    for i in range(len(top)):
        ranks[np.where(top[i] == 1)[0]] = i
    return ranks, all_best_hand_player

@nb.njit()
def showdown(env_state):
    # print('before showdown', env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE])
    # board = env_state[ENV_CARD_OPEN : ENV_ALL_PLAYER_CHIP]
    chip_give = env_state[ENV_ALL_PLAYER_CHIP_IN_POT : ENV_ALL_PLAYER_STATUS]
    # print('give', chip_give)
    status = env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD]
    # print('status', status)
    # hands = [np.array([env_state[ENV_ALL_FIRST_CARD + i ], env_state[ENV_ALL_SECOND_CARD + i]]) for i in range(9)]
    player_chip_receive = np.zeros(9)
    all_pot_val = np.sum(chip_give)
    if np.count_nonzero(status) == 1:
        player_chip_receive[np.argmax(status)] = all_pot_val
        #update chip sau ván chơi của các người chơi
        env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE] += player_chip_receive
    else:
        board = env_state[ENV_CARD_OPEN : ENV_ALL_PLAYER_CHIP]
        hands = [np.array([env_state[ENV_ALL_FIRST_CARD + i ], env_state[ENV_ALL_SECOND_CARD + i]]) for i in range(9)]

        #tính rank bài của các người chơi
        ranks, all_player_hand = holdem(board, hands)
        # print('ranks', ranks, 'status', status)
        #split_pot:
        max_chip_can_get = np.zeros(9)
        while all_pot_val > 0:
            #tạo side pot
            side_pot = np.min(chip_give[chip_give > 0])
            player_join = np.where(chip_give >= side_pot)[0]
            player_get_pot = np.where((chip_give >= side_pot) & (status == 1) )
            side_pot_val = len(player_join)*side_pot
            max_chip_can_get[player_get_pot] += side_pot_val
            #lấy rank
            rank_in_side_pot = np.full(9,10)
            rank_in_side_pot[player_get_pot] = ranks[player_get_pot]

            best_rank = np.min(rank_in_side_pot)
            # print('rank', best_rank, player_get_pot, side_pot)
            player_win_pot = np.where(rank_in_side_pot == best_rank)[0]
            delta_chip = int(side_pot_val/len(player_win_pot))
            player_chip_receive[player_win_pot] += delta_chip
            #cập nhật chip còn lại
            all_pot_val -= side_pot_val
            temp = chip_give - side_pot
            chip_give = (temp)*(temp > 0)
            # print(all_pot_val,side_pot,player_chip, delta_chip)
        #update chip sau ván chơi của các người 
        env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE] += player_chip_receive

        #show card on hand, update vào env_state xem ai phải show bài, lá nào k cần show thì gán thành -1
        rank_temp = np.full(9, 10)
        chip_receive_max = np.zeros(9)
        #lấy rank hand người mở đầu tiên và lượng chip tối đa họ có thể ăn
        rank_temp[int(env_state[ENV_TEMP_BUTTON])] = ranks[int(env_state[ENV_TEMP_BUTTON])] 
        chip_receive_max[int(env_state[ENV_TEMP_BUTTON])] = max_chip_can_get[int(env_state[ENV_TEMP_BUTTON])]
        for i in range(1, 9):
            id = int(env_state[ENV_TEMP_BUTTON] + i ) % 9
            #nếu người chơi đã bỏ, thì bài của người chơi là bài trên bàn
            if status[id] == 0:
                # env_state[ENV_ALL_FIRST_CARD_SHOWDOWN + id] = -1
                # env_state[ENV_ALL_FIRST_CARD_SHOWDOWN + id] = -1
                continue
            #nếu người chơi bài cao hơn hoặc bằng (share pot) hoặc có thể ăn side pot thì phải show card
            if np.min(rank_temp) >= ranks[id] or np.max(chip_receive_max[np.where(rank_temp < ranks[id])[0]]) < max_chip_can_get[id]:
                if env_state[ENV_ALL_FIRST_CARD + id] in all_player_hand[id]:
                    env_state[ENV_ALL_FIRST_CARD_SHOWDOWN + id] = env_state[ENV_ALL_FIRST_CARD + id]
                if env_state[ENV_ALL_SECOND_CARD + id] in all_player_hand[id]:
                    env_state[ENV_ALL_SECOND_CARD_SHOWDOWN + id] = env_state[ENV_ALL_SECOND_CARD + id]
                rank_temp[id] = ranks[id]
                chip_receive_max[id] = max_chip_can_get[id]
            # else:
            #     # env_state[ENV_ALL_FIRST_CARD_SHOWDOWN + id] = -1
                # env_state[ENV_ALL_FIRST_CARD_SHOWDOWN + id] = -1
    # print('receive', player_chip_receive.astype(np.int64))
    # print('resu;t', env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE].astype(np.int64), np.sum(env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE]))
    # if np.sum(env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE]) > 1800:
    #     raise Exception('toang tổng')
    return env_state

def action_player(env_state,list_player,file_temp,file_per):
    current_player = int(env_state[ENV_ID_ACTION])
    player_state = getAgentState(env_state)
    played_move,file_temp[current_player],file_per = list_player[current_player](player_state,file_temp[current_player],file_per)
    # print(current_player, getValidActions(player_state))
    if getValidActions(player_state)[played_move] != 1:
        raise Exception('bot dua ra action khong hop le')
    return played_move,file_temp,file_per

@nb.njit()
def getActionSize():
    return 6

@nb.njit()
def getAgentSize():
    return 9

@nb.njit()
def getStateSize():
    return PLAYER_STATE_LENGTH

@nb.njit()
def checkEnded(env_state):
    all_player_chip = env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE]
    all_player_status = env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD]
    number_game = env_state[ENV_NUMBER_GAME_PLAYED]
    if (np.count_nonzero(all_player_status) == 1 and np.count_nonzero(all_player_chip) == 1) or number_game == 200:
        # print('toang', all_player_chip, all_player_status, number_game)
        return True
    return False

@nb.njit()
def getReward(agent_state):
    all_player_chip = agent_state[P_ALL_PLAYER_CHIP : P_ALL_PLAYER_CHIP_GIVE]
    all_player_status = agent_state[P_ALL_PLAYER_STATUS : P_NUMBER_GAME_PLAY]

    if (np.count_nonzero(all_player_status) == 1 and np.count_nonzero(all_player_chip) == 1) or agent_state[P_NUMBER_GAME_PLAY] == 200:
        if np.argmax(all_player_chip) == 0:
            return 1
        else:
            return 0
    else:
        return -1

def player_random(player_state, file_temp, file_per):
    #print(player_state[P_ALL_PLAYER_CHIP], player_state[P_CASH_TO_CALL], player_state[P_CASH_TO_BET])
    binary_action  = getValidActions(player_state)
    # if np.random.rand() < 0.998:
    #     # binary_action[3] = 0
    #     binary_action[4] = 0
    # if np.random.rand() < 0.5:
    #     binary_action[3] = 0

    
    #print(binary_action)
    list_action = np.where(binary_action == 1)[0]
    action = int(np.random.choice(list_action))
    if getReward(player_state) == -1:
        # #print('chưa hết game')
        pass
    else:
        if getReward(player_state) == 1:
            #print(player_state[P_ALL_PLAYER_STATUS:P_BUTTON_DEALER])
            #print('win')
            pass
        else:
            #print('lose')
            pass
    # print(list_action)
    return action, file_temp, file_per

def player_input(player_state, file_temp, file_per):
    #print(player_state[P_ALL_PLAYER_CHIP], player_state[P_CASH_TO_CALL], player_state[P_CASH_TO_BET])
    binary_action  = getValidActions(player_state)
    list_action = np.where(binary_action == 1)[0]
    action = int(input(f'action possible {list_action}'))
    if getReward(player_state) == -1:
        # print('chưa hết game')
        pass
    else:
        if getReward(player_state) == 1:
            #print(player_state[P_ALL_PLAYER_STATUS:P_BUTTON_DEALER])
            #print('win')
            pass
        else:
            #print('lose')
            pass
    # print(list_action)
    return action, file_temp, file_per

@nb.njit()
def check_winner(env_state):
    all_player_chip = env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE]
    winner = np.argmax(all_player_chip)
    return winner

def normal_main(list_player, times, file_per):
    count = np.zeros(len(list_player)+1)
    all_id_player = np.arange(len(list_player))
    for van in range(times):
        shuffle = np.random.choice(all_id_player, NUMBER_PLAYER, replace=False)
        shuffle_player = [list_player[shuffle[i]] for i in range(NUMBER_PLAYER)]
        file_temp = [[0]]*NUMBER_PLAYER
        winner, file_per = one_game(shuffle_player, file_temp, file_per)
        if winner == -1:
            count[winner] += 1
        else:
            count[shuffle[winner]] += 1
    return list(count.astype(np.int64)), file_per


def one_game(list_player, file_temp, file_per):
    env_state = initEnv()
    count_game = 0
    while not checkEnded(env_state):
        env_state = reset_round(env_state)
        # print('bài chung: ', ALL_CARD_STR[env_state[ENV_CARD_OPEN : ENV_ALL_PLAYER_CHIP].astype(np.int64)])
        # print('bài riêng1: ',  ALL_CARD_STR[env_state[ENV_ALL_FIRST_CARD: ENV_ALL_SECOND_CARD].astype(np.int64)])
        # print('bài riêng2: ',  ALL_CARD_STR[env_state[ENV_ALL_SECOND_CARD: ENV_ALL_FIRST_CARD_SHOWDOWN].astype(np.int64)])

        while env_state[ENV_STATUS_GAME] != 6:
            action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
            env_state = stepEnv(env_state, action)
        # print('turn bonus++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        for id in range(NUMBER_PLAYER):
            id_player = int(id + env_state[ENV_TEMP_BUTTON]) % NUMBER_PLAYER
            if env_state[ENV_ALL_PLAYER_STATUS + id_player] == 0:
                continue
            else:
                env_state[ENV_ID_ACTION] = id_player
                action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
        # print('end turn bonus+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        count_game += 1
    winner = check_winner(env_state)
    for id_player in range(NUMBER_PLAYER):
        env_state[ENV_ID_ACTION] = id_player
        action, file_temp, file_per = action_player(env_state,list_player,file_temp,file_per)
    return winner, file_per



