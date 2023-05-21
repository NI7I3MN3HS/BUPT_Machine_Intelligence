# 交换
def swap_char(a, i, j):
    if i > j:
        i, j = j, i
    # 得到ij交换后的数组
    b = a[:i] + a[j] + a[i + 1: j] + a[i] + a[j + 1:]
    return b


# 启发式函数，曼哈顿距离
def hn(current, target):
    sum = 0
    a = current.index("0")
    for i in range(0, 9):
        if i != a:
            sum = sum + abs(i - target.index(current[i]))
    return sum


# A*算法
def A_star(start, target):
    # 先进行判断初始状态和目标状态逆序值是否同是奇数或偶数
    start_inversion = 0  # 初始状态逆序值
    target_inversion = 0  # 目标状态逆序值

    for i in range(1, 9):
        for j in range(0, i):
            if start[j] > start[i] and start[i] != "0":  # 0是false,'0'才是数字
                start_inversion = start_inversion + 1
    for i in range(1, 9):
        for j in range(0, i):
            if target[j] > target[i] and target[i] != "0":
                target_inversion = target_inversion + 1
    if (start_inversion % 2) != (target_inversion % 2):  # 一个奇数一个偶数，不可达
        return -1, None

    dict_position = {}  # 记录每个节点的父节点 也是记录每个节点是否被遍历过 closed表
    dict_position_gn = {}  # 记录每个节点的gn值
    dict_position_fn = {}  # 记录每个节点的fn值 也是open表
    # 每个位置可交换的位置集合
    dict_shifts = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4, 6],
        4: [1, 3, 5, 7],
        5: [2, 4, 8],
        6: [3, 7],
        7: [4, 6, 8],
        8: [5, 7],
    }

    dict_position[start] = -1
    dict_position_gn[start] = 0
    dict_position_fn[start] = dict_position_gn[start] + hn(start, target)

    while len(dict_position_fn) > 0:
        current = min(dict_position_fn, key=dict_position_fn.get)
        del dict_position_fn[current]  # 当前节点从open表移除
        if current == target:  # 判断当前状态是否为目标状态
            break
        # 寻找0的位置。
        zero_postion = current.index("0")
        shifts_position = dict_shifts[zero_postion]  # 当前可进行交换的位置集合
        for i in shifts_position:
            new = swap_char(current, zero_postion, i)
            if dict_position.get(new) == None:
                dict_position_gn[new] = dict_position_gn[current] + 1
                dict_position_fn[new] = dict_position_gn[new] + hn(new, target)
                dict_position[new] = current

    steps = []  # 路径列表
    steps.append(current)

    while dict_position[current] != -1:  # 存入路径
        current = dict_position[current]
        steps.append(current)
    steps.reverse()

    return 0, steps


if __name__ == "__main__":
    # 测试数据
    start = "013425768"  # 初始状态
    target = "647850321"  # 目标状态
    start = "013425786"  # 初始状态
    target = "647850321"  # 目标状态

    is_solve, steps = A_star(start, target)
    if is_solve != 0:
        print("无解")
    else:
        for i in range(len(steps)):
            print("step #" + str(i + 1))
            print(steps[i][:3])
            print(steps[i][3:6])
            print(steps[i][6:])
