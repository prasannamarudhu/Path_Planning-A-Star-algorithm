############################################################################

# *********** PRASANNA MARUDHU BALASUBRAMANIAN  ***********#
# *********** UID: 116197700  ***********#
# *********** ENPM661 - Planning for Autonomous Robots ****#
# *********** Class : Spring 2019  ***********#
# *********** Part2 : A-Star for Rigid Robot  ***********#

# Comments
# The below are the modules used in this project
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue


# This function creates the obstacle space in the Map
def create_graph(rrad, clrnc, resol):
    # Obstacle space defining
    r = 15 / resol
    a = 15 / resol
    b = 6 / resol

    # Rectangle Obstacle
    A1 = [-1, 1, 0, 0]
    B1 = [0, 0, -1, 1]
    C1 = [(round(50 / resol)), (round(-100 / resol)), (round(67.5 / resol)), (round(-112.5 / resol))]

    # Irregular Polygon Obstacle
    xy = np.array([[(round(125 / resol)), (round(56 / resol))], [(round(150 / resol)), (round(15 / resol))],
                   [(round(173 / resol)), (round(15 / resol))], [(round(163 / resol)), (round(52 / resol))],
                   [(round(170 / resol)), (round(90 / resol))], [(round(193 / resol)), (round(52 / resol))]])

    # Part 1
    m1 = (xy[1][1] - xy[0][1]) / (xy[1][0] - xy[0][0])  # Slope
    c1 = xy[0][1] - ((m1) * (xy[0][0]))  # Y- Intercept.

    m2 = (xy[2][1] - xy[1][1]) / (xy[2][0] - xy[1][0])
    c2 = xy[1][1] - m2 * xy[1][0]

    m3 = (xy[3][1] - xy[2][1]) / (xy[3][0] - xy[2][0])
    c3 = xy[2][1] - m3 * xy[2][0]

    m4 = (xy[0][1] - xy[3][1]) / (xy[0][0] - xy[3][0])
    c4 = xy[3][1] - m4 * xy[3][0]

    A2 = [m1, m2, -m3, -m4]
    B2 = [-1, -1, 1, 1]
    C2 = [c1, c2, -c3, -c4]

    # Part 2

    m5 = (xy[5][1] - xy[4][1]) / (xy[5][0] - xy[4][0])  # Slope Values
    c5 = xy[4][1] - m5 * xy[4][0]  # Y -Intercept

    m6 = (xy[4][1] - xy[3][1]) / (xy[4][0] - xy[3][0])
    c6 = xy[3][1] - m6 * xy[3][0]

    m7 = (xy[3][1] - xy[2][1]) / (xy[3][0] - xy[2][0])
    c7 = xy[2][1] - m7 * xy[2][0]

    m8 = (xy[2][1] - xy[5][1]) / (xy[2][0] - xy[5][0])
    c8 = xy[5][1] - m8 * xy[5][0]

    A3 = [-m5, - m6, m7, m8]
    B3 = [1, 1, - 1, - 1]
    C3 = [-c5, - c6, c7, c8]

    # Defining the obstacle graph
    graph = {}

    ########
    # Radius of the Robot and clearance

    Rob_Clr = rrad + clrnc

    Cr1 = [None] * 4
    Cr2 = [None] * 4
    Cr3 = [None] * 4

    for i in range(0, 4):
        Cr1[i] = (C1[i]) - (Rob_Clr)
        Cr2[i] = (C2[i]) - (Rob_Clr)
        Cr3[i] = (C3[i]) - (Rob_Clr)
    ######

    for i in range(round(250 / resol)):
        for j in range(round(150 / resol)):
            graph[(i, j)] = {'visited': False, 'distance': np.inf, 'valid': True, 'parent': (0, 0)}

            if ((i - (round(190 / resol))) ** 2 + (j - (round(130 / resol))) ** 2 - (r + Rob_Clr) ** 2) <= 0:
                graph[(i, j)]['valid'] = False
            if (((i - (round(140 / resol))) ** 2) / ((a + Rob_Clr) ** 2) + ((j - (round(120 / resol))) ** 2) / (
                    (b + Rob_Clr) ** 2) - 1) <= 0:
                graph[(i, j)]['valid'] = False
            if ((A1[0] * i + B1[0] * j + Cr1[0]) <= 0) and ((A1[1] * i + B1[1] * j + Cr1[1]) <= 0) and (
                    (A1[2] * i + B1[2] * j + Cr1[2]) <= 0) and ((A1[3] * i + B1[3] * j + Cr1[3]) <= 0):
                graph[(i, j)]['valid'] = False
            if (A2[0] * i + B2[0] * j + Cr2[0] <= 0) and (A2[1] * i + B2[1] * j + Cr2[1] <= 0) and (
                    A2[2] * i + B2[2] * j + Cr2[2] <= 0) and (A2[3] * i + B2[3] * j + Cr2[3] <= 0):
                graph[(i, j)]['valid'] = False
            if (A3[0] * i + B3[0] * j + Cr3[0] <= 0) and (A3[1] * i + B3[1] * j + Cr3[1] <= 0) and (
                    A3[2] * i + B3[2] * j + Cr3[2] <= 0) and (A3[3] * i + B3[3] * j + Cr3[3] <= 0):
                graph[(i, j)]['valid'] = False

    return graph


# This function generates unique id for each node which enable us to search for the availability of the node faster
# Quicker operation is achieved with this function
def ID_Generation(x, y):
    return y * 120 + x


# Heuristic Distance Calculation
def calculate_distance(goal, current):
    w = 1.0  # heuristic weight is 1(non-weighted A-Star)
    d = w * math.sqrt(
        ((goal[0] - current[0]) * (goal[0] - current[0])) + ((goal[1] - current[1]) * (goal[1] - current[1])))
    return d


# Defininf the Class Algorithm_PathCalcNode and initializing the values in the class
class Algorithm_PathCalcNode:

    def __init__(self, x, y, cost, Node_id, Parent_id, heuristic_cost):
        self.id = Node_id
        self.parent_id = Parent_id
        self.Heuristic_cost = heuristic_cost
        self.x = x
        self.cost = cost
        self.y = y


# Calculating the optimal path based on the parent nodes saved earlier
def Path_Calculation(node_setclosed, goal_node):
    Path_x = [goal_node.x]
    Path_y = [goal_node.y]
    p_id = goal_node.parent_id

    while p_id != -1:
        node = node_setclosed[p_id]
        Path_x.append(node.x)
        Path_y.append(node.y)
        p_id = node.parent_id

    return Path_x, Path_y


# This function defines the A-Star logic
def A_STAR(start, resol_grid, goal, points):
    # Creating the dictionaries
    data_setopen = dict()
    node_setclosed = dict()
    points_2plot = list()

    # Plotting the start and goal points
    plt.plot(start[0], start[1], "rx")
    plt.plot(goal[0], goal[1], "rx")

    # Calculating the cost
    cost_heuristic = Distance_Heuristic(start[0], start[1], goal[0], goal[1])

    # Obtaining the initialized values from the class
    source_node = Algorithm_PathCalcNode(start[0], start[1], 0, 0, -1, cost_heuristic)
    goal_node = Algorithm_PathCalcNode(goal[0], goal[1], 0, 0, 0, 0)

    # Obtaining the ID of the nodes
    source_node.id = ID_Generation(source_node.x, source_node.y)
    data_setopen[source_node.id] = source_node

    while len(data_setopen) != 0:

        # Choosing the Current node ID as the node with the minimium heuristic cost
        Node_Current_ID = min(data_setopen, key=lambda i: data_setopen[i].Heuristic_cost)

        # Obtaining the node values belonging to the ID chosen
        Node_Current = data_setopen[Node_Current_ID]

        # Appending the node values
        points_2plot.append((Node_Current.x, Node_Current.y))
        x, y = zip(*points_2plot)

        # Exiting the function once the goal node is reached
        if Node_Current.x == goal_node.x and Node_Current.y == goal_node.y:
            goal_node.parent_id = Node_Current.parent_id
            goal_node.cost = Node_Current.cost
            plt.plot(x, y, "g.")
            plt.grid(True)
            plt.axis("equal")
            plt.grid(b=True, which='major', color='y', linestyle='-')
            plt.grid(b=True, which='minor', color='y', linestyle='-', alpha=0.2)
            plt.minorticks_on()
            plt.xlim(0, round(250 / resol_grid))
            plt.ylim(0, round(150 / resol_grid))

            break
        # Plotting the generated nodes for every 50th node
        if len(points_2plot) % (50 / resol_grid) == 0:
            plt.plot(x, y, "g.")
            plt.grid(True)
            plt.axis("equal")
            plt.grid(b=True, which='major', color='y', linestyle='-')
            plt.grid(b=True, which='minor', color='y', linestyle='-', alpha=0.2)
            plt.minorticks_on()
            plt.xlim(0, round(250 / resol_grid))
            plt.ylim(0, round(150 / resol_grid))
            plt.pause(0.0001)
            points_2plot.clear()

        # Moving the current node id into the set of closed nodes
        del data_setopen[Node_Current_ID]
        node_setclosed[Node_Current.id] = Node_Current

        # Creating the new neighbor nodes
        Node_Neighbor = Neighbor_NodeCreate(Node_Current, goal_node, points, resol_grid)

        for Neighbor in Node_Neighbor:
            # Continuing if the neighbor id is already in the closed set of nodes
            if Neighbor.id in node_setclosed:
                continue
            # On the other hand if the neighbor id is in the open set then checking for the
            # node which has the lowest cost and assigning it as the parent node
            if Neighbor.id in data_setopen:
                if data_setopen[Neighbor.id].cost > Neighbor.cost:
                    data_setopen[Neighbor.id].cost = Neighbor.cost
                    data_setopen[Neighbor.id].Heuristic_cost = Neighbor.Heuristic_cost
                    data_setopen[Neighbor.id].parent_id = Node_Current_ID
            # Otherwise assigning this neighbor node into the opendata set
            else:
                data_setopen[Neighbor.id] = Neighbor
    # Finding the x and y values for the optimal path to be plotted
    Path_x, Path_y = Path_Calculation(node_setclosed, goal_node)
    for i in range(len(Path_x)):
        plt.plot(Path_x[i], Path_y[i], "b.")

    # x = [i[0] for i in Path_x]
    # y = [i[0] for i in Path_y]
    plt.plot(Path_x, Path_y, 'r-')
    plt.grid(True)
    plt.axis("equal")
    plt.grid(b=True, which='major', color='y', linestyle='-')
    plt.grid(b=True, which='minor', color='y', linestyle='-')
    plt.minorticks_on()
    plt.xlim(0, round(250 / resol_grid))
    plt.ylim(0, round(150 / resol_grid))
    plt.show()


# This function creates the neighbor nodes
def Neighbor_NodeCreate(Node_Current, goal_node, obstacle_points, resol_grid):
    X = Node_Current.x
    Y = Node_Current.y

    # Possible moves for each node around them
    move = [(1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1), (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

    # Assignin the values
    Goal_X = goal_node.x
    Goal_Y = goal_node.y
    cost = Node_Current.cost
    Node_Neighbor = []

    # Creating the nodes for all possible 8 moves
    for i in range(0, 8):

        cost_new = cost + move[i][2]
        Gen_X = X + move[i][0]
        Gen_Y = Y + move[i][1]
        # Calculating the heuristic cost for each possible nodes
        Heuristic_cost = cost_new + Distance_Heuristic(Gen_X, Gen_Y, Goal_X,
                                                       Goal_Y)  # Calculating the heuristic cost for each possible nodes
        Pos = (Gen_X, Gen_Y)
        # Checking whether the new node generated is in the obstacle space or not
        if Gen_X in range(0, round(251 / resol_grid)) and Gen_Y in range(0, round(151 / resol_grid)):
            if Pos not in obstacle_points:
                # If the node is not in the obstacle space the generating an ID of this node
                ID = ID_Generation(Gen_X, Gen_Y)
                neighbor_node = Algorithm_PathCalcNode(Gen_X, Gen_Y, cost_new, ID, Node_Current.id, Heuristic_cost)
                # Appending the generated node into node set
                Node_Neighbor.append(neighbor_node)

    return Node_Neighbor


# This function accepts the user input and assign to the variables
def user_input():
    print("Please feed the Following Information for Path Planning ")

    sx = input("Enter the Start node 'X' value : ")
    sy = input("Enter the Start node 'Y' value : ")
    rrad = input("Enter the Radius of robot value : ")
    clrnc = input("Enter the clearance value : ")
    resol = input("Enter the grid resolution value : ")
    gx = input("Enter the Goal node 'X' value : ")
    gy = input("Enter the Goal node 'Y' value : ")

    return int(sx), int(sy), int(rrad), int(clrnc), int(resol), int(gx), int(gy)
    # return sx, sy, rrad, clrnc, resol, gx, gy


# Heuristic Calcualtion
def Distance_Heuristic(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


######## Main Function #####
if __name__ == "__main__":

    sx, sy, rrad, clrnc, resol, gx, gy = user_input()
    sx = (round(sx / resol))
    sy = (round(sy / resol))
    gx = (round(gx / resol))
    gy = (round(gy / resol))
    rrad = (round(rrad / resol))
    clrnc = (round(clrnc / resol))

    if sx < 0 or sy < 0 or gx < 0 or gy < 0 or clrnc < 0 or rrad < 0:
        print("\n\nNegative values for node positions and other parameters not allowed")
        exit()

    g = create_graph(rrad, clrnc, resol)

    points = [x for x in g.keys() if not (g[x]['valid'])]
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    plt.figure(figsize=(15, 15))
    ax = plt.axes()

    plt.xlabel('x')
    plt.ylabel('y')

    ax.set_xlim(0, (round(250 / resol)))
    ax.set_ylim(0, (round(150 / resol)))

    plt.plot(x, y, ".k")
    plt.title("A-Star Simulation for Rigid Robot")

    # Checking whether the input node points are in the obstacle space
    input = (sx, sy)  # start node points
    #if g[sx, sy]['valid'] == True and g[gx, gy]['valid'] == True and sx <= (round(250 / resol)) and sy <= (round(150 / resol)) and gx <= (round(250 / resol)) and gy <= (round(150 / resol)):

    A_STAR((sx, sy), resol, (gx, gy), points)


    # Asking the user to give some other inpout points since they are not Valid
    """
    else:
        print(
            " \n*****The start and goal node position values you entered is inside obstacle space or beyond the map boundary*****\n*****Please enter other start and goal node position when you run again*****")
        print(" \n********* Terminating Project Execution now ********* ")
        exit()
    """
############