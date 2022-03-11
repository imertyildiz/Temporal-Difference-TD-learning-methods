import random


def max_action(environment, state, x, y, obsclate_states):
    max_value = - float('inf')
    max_action = None
    actions = ["<", "^", ">", "V"]
    for action in actions:
        if action == "<":
            if state[1] == 0 or ((state[0], state[1] - 1) in obsclate_states):
                #  take state
                if environment[state] > max_value:
                    max_value = environment[state]
                    max_action = action
            else:
                #  take action
                if environment[state[0], state[1] - 1] > max_value:
                    max_value = environment[state[0], state[1] - 1]
                    max_action = action
        elif action == "^":
            if state[0] == 0 or ((state[0] - 1, state[1]) in obsclate_states):
                #  take state
                if environment[state] > max_value:
                    max_value = environment[state]
                    max_action = action
            else:
                #  take action
                if environment[state[0] - 1, state[1]] > max_value:
                    max_value = environment[state[0] - 1, state[1]]
                    max_action = action
        elif action == ">":
            if state[1] == y or ((state[0], state[1] + 1) in obsclate_states):
                #  take state
                if environment[state] > max_value:
                    max_value = environment[state]
                    max_action = action
            else:
                #  take action
                if environment[state[0], state[1] + 1] > max_value:
                    max_value = environment[state[0], state[1] + 1]
                    max_action = action
        else:
            if state[0] == x or ((state[0] + 1, state[1]) in obsclate_states):
                #  take state
                if environment[state] > max_value:
                    max_value = environment[state]
                    max_action = action
            else:
                #  take action
                if environment[state[0] + 1, state[1]] > max_value:
                    max_value = environment[state[0] + 1, state[1]]
                    max_action = action
    return max_action


def max_action_Q_learning(environment, state, x, y, obsclate_states):
    max_value = - float('inf')
    max_action = None
    actions = ["<", "^", ">", "V"]
    for action in actions:
        if action == "<":
            #  take state
            if max_value < environment[state, "<"]:
                max_value = environment[state, "<"]
                max_action = action
        elif action == "^":
            #  take state
            if max_value < environment[state, "^"]:
                max_value = environment[state, "^"]
                max_action = action
        elif action == ">":
            #  take state
            if max_value < environment[state, ">"]:
                max_value = environment[state, ">"]
                max_action = action
        else:
            #  take state
            if max_value < environment[state, "V"]:
                max_value = environment[state, "V"]
                max_action = action
    return max_action


def get_next_state(environment, state, x, y, obsclate_states, action):
    if action == "<":
        if state[1] == 0 or ((state[0], state[1] - 1) in obsclate_states):
            return state
        else:
            return (state[0], state[1] - 1)
    elif action == "^":
        if state[0] == 0 or ((state[0] - 1, state[1]) in obsclate_states):
            return state
        else:
            return (state[0] - 1, state[1])
    elif action == ">":
        if state[1] == y or ((state[0], state[1] + 1) in obsclate_states):
            return state
        else:
            return (state[0], state[1] + 1)
    else:
        if state[0] == x or ((state[0] + 1, state[1]) in obsclate_states):
            return state
        else:
            return (state[0] + 1, state[1])


def TD_0_implementation(environment, start_state, reward, learning_rate, gamma, episode_count, epsilon, goal_states,
                        action_noise, x, y, obsclate_states):
    policy = {}
    for i in environment:
        if (i not in goal_states) and (i not in obsclate_states):
            policy[i] = ""
    for i in range(0, episode_count):
        current_state = start_state
        while current_state not in goal_states:
            selected_value = None
            selected_action = None
            actions = ["<", "^", ">", "V"]
            if random.random() <= epsilon:
                actions = ["<", "^", ">", "V"]
                selected_action = actions[random.randint(0, 3)]
            else:
                selected_action = max_action(environment, current_state, x, y, obsclate_states)

            action_list = [actions[actions.index(selected_action) - 1], actions[actions.index(selected_action)]]
            if actions.index(selected_action) == 3:
                action_list.append(actions[0])
            else:
                action_list.append(actions[actions.index(selected_action) + 1])
            # bak
            selected_action = random.choices(action_list, action_noise)[0]
            next_state = get_next_state(environment, current_state, x, y, obsclate_states, selected_action)
            # policy[current_state] = selected_action
            # policy extraction
            if next_state in goal_states:
                # TODO: Neden utility of goal state ?
                rewardx = reward + goal_states[next_state]
                environment[current_state] = environment[current_state] + learning_rate * (
                        rewardx + gamma * environment[next_state] - environment[current_state])
                break
            else:
                environment[current_state] = environment[current_state] + learning_rate * (
                        reward + gamma * environment[next_state] - environment[current_state])
            # TODO: Next state olmali mi ?
            current_state = next_state

    for i in obsclate_states:
        environment.pop(i)
    for i in goal_states:
        environment[i] = goal_states[i]
    for i in range(x + 1):
        for j in range(y + 1):
            if (i, j) not in obsclate_states and (i, j) not in goal_states:
                policy[(i, j)] = max_action(environment, (i, j), x, y, obsclate_states)
    for i in environment:
        environment[i] = round(environment[i], 2)
    return environment, policy


def Q_learning_implementation(environment_inital, start_state, reward, learning_rate, gamma, episode_count, epsilon,
                              goal_states,
                              action_noise, x, y, obsclate_states):
    policy = {}
    for i in environment_inital:
        if (i not in goal_states) and (i not in obsclate_states):
            policy[i] = ""
    environment_real = {}
    for i in environment_inital:
        for j in ["<", "^", ">", "V"]:
            environment_real[i, j] = 0.0

    for i in range(0, episode_count):
        current_state = start_state
        while current_state not in goal_states:
            selected_value = None
            selected_action = None
            actions = ["<", "^", ">", "V"]
            if random.random() <= epsilon:
                actions = ["<", "^", ">", "V"]
                selected_action = actions[random.randint(0, 3)]
            else:
                selected_action = max_action_Q_learning(environment_real, current_state, x, y, obsclate_states)

            action_list = [actions[actions.index(selected_action) - 1], actions[actions.index(selected_action)]]
            if actions.index(selected_action) == 3:
                action_list.append(actions[0])
            else:
                action_list.append(actions[actions.index(selected_action) + 1])
            # bak
            selected_action_new = random.choices(action_list, action_noise)[0]
            next_state = get_next_state(environment_real, current_state, x, y, obsclate_states, selected_action_new)
            # policy[current_state] = selected_action
            # policy extraction
            if next_state in goal_states:
                # TODO: Neden utility of goal state ?
                rewardx = reward + goal_states[next_state]
                environment_real[current_state, selected_action] = environment_real[
                                                                       current_state, selected_action] + learning_rate * (
                                                                           rewardx + gamma * max(
                                                                       environment_real[next_state, "<"],
                                                                       environment_real[next_state, "^"],
                                                                       environment_real[next_state, ">"],
                                                                       environment_real[next_state, "V"]) -
                                                                           environment_real[
                                                                               current_state, selected_action])
                break
            else:
                environment_real[current_state, selected_action] = environment_real[
                                                                       current_state, selected_action] + learning_rate * (
                                                                           reward + gamma * max(
                                                                       environment_real[next_state, "<"],
                                                                       environment_real[next_state, "^"],
                                                                       environment_real[next_state, ">"],
                                                                       environment_real[next_state, "V"]) -
                                                                           environment_real[
                                                                               current_state, selected_action])
            # TODO: Next state olmali mi ?
            current_state = next_state

    for i in obsclate_states:
        for j in ["<", "^", ">", "V"]:
            environment_real.pop(i, j)
    for i in goal_states:
        for j in ["<", "^", ">", "V"]:
            environment_real[i, j] = goal_states[i]
    for i in range(x + 1):
        for j in range(y + 1):
            if (i, j) not in obsclate_states and (i, j) not in goal_states:
                policy[(i, j)] = max_action_Q_learning(environment_real, (i, j), x, y, obsclate_states)
    for i in environment_inital:
        environment_inital[i] = max(
            environment_real[i, "<"],
            environment_real[i, "^"],
            environment_real[i, ">"],
            environment_real[i, "V"])
    for i in obsclate_states:
        environment_inital.pop(i)
    for i in goal_states:
        environment_inital.pop(i)
    for i in environment_inital:
        environment_inital[i] = round(environment_inital[i], 2)
    return environment_inital, policy


def SolveMDP(method_name, problem_file_name, random_seed):
    with open(problem_file_name) as f:
        lines = f.read().splitlines()
    random.seed(random_seed)
    environment = {}
    lines.pop(0)  # [environment]
    size = lines.pop(0).split(" ")
    size_x = int(size[0])
    size_y = int(size[1])
    for i in range(0, size_x):
        for j in range(0, size_y):
            environment[(i, j)] = 0.0
    lines.pop(0)  # [obstacle states]
    line = lines.pop(0).split("|")
    obsclate_states = []
    for i in line:
        x = i.replace("(", "").replace(")", "").split(",")
        obsclate_states.append(tuple((int(x[0]), int(x[1]))))
    lines.pop(0)
    line = lines.pop(0).split("|")
    goal_states = {}
    for i in line:
        y = i.split(":")
        x = y[0].replace("(", "").replace(")", "").split(",")
        # environment[(int(x[0]), int(x[1]))] = float(y[1])
        goal_states[tuple((int(x[0]), int(x[1])))] = float(y[1])
    lines.pop(0)  # start state
    start_initial = lines.pop(0)
    start_initial = start_initial.replace("(", "").replace(")", "").split(",")
    start_state = tuple((int(start_initial[0]), int(start_initial[1])))
    lines.pop(0)  # Reward
    reward = float(lines.pop(0))
    lines.pop(0)
    action_noise = []
    x_ = float(lines.pop(0))
    action_noise.append(float(lines.pop(0)))  # b - a - c -> a-b-c
    action_noise.append(x_)  # b - a - c -> a-b-c
    action_noise.append(float(lines.pop(0)))  # b - a - c -> a-b-c
    lines.pop(0)  # Learning Rate
    learning_rate = float(lines.pop(0))
    lines.pop(0)  # gamma
    gamma = float(lines.pop(0))
    lines.pop(0)  # epsilon
    epsilon = float(lines.pop(0))
    lines.pop(0)  # episode count
    episode_count = int(lines.pop(0))
    if method_name == "TD(0)":
        return TD_0_implementation(environment, start_state, reward, learning_rate, gamma, episode_count, epsilon,
                                   goal_states, action_noise, size_x - 1,
                                   size_y - 1, obsclate_states)
    elif method_name == "Q-learning":
        return Q_learning_implementation(environment, start_state, reward, learning_rate, gamma, episode_count, epsilon,
                                         goal_states, action_noise, size_x - 1,
                                         size_y - 1, obsclate_states)

