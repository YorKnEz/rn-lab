import re
import math
import copy

def parse(filename):
    a = []
    b = []

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            [first, second] = line.split(" = ")
            b.append(int(second))

            coefs_str = re.findall(r"([+-] )?(\d*[xyz])", first)

            coefs = []

            for sign, var in coefs_str:
                coef = var[:-1]

                coef = 1 if len(coef) == 0 else int(coef)

                if len(sign) > 0 and sign[0] == '-':
                    coef = -coef

                coefs.append(coef)

            a.append(coefs)

    return (a, b)

def det(matrix):
    return \
        matrix[0][0] * matrix[1][1] * matrix[2][2] + \
        matrix[0][1] * matrix[1][2] * matrix[2][0] + \
        matrix[1][0] * matrix[2][1] * matrix[0][2] - \
        matrix[0][2] * matrix[1][1] * matrix[2][0] - \
        matrix[0][1] * matrix[1][0] * matrix[2][2] - \
        matrix[0][0] * matrix[1][2] * matrix[2][1]

def trace(matrix):
    return matrix[0][0] + matrix[1][1] + matrix[2][2]

def transpose(matrix):
    t = [[], [], []]

    for line in matrix:
        for j, x in enumerate(line):
            t[j].append(x)

    return t

def norm(vec):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

def dot(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]

def mul_mv(matrix, vec):
    r = []

    for line in matrix:
        r.append(dot(line, vec))

    return r

def solve_cramer(a, b):
    sol = []
    d = det(a)

    for i in range(3):
        m = copy.deepcopy(a)

        for j in range(3):
            m[j][i] = b[j]

        sol.append(det(m) / d)

    return sol

def cofactor(matrix):
    m = []

    for i in range(3):
        m.append([])

        for j in range(3):
            # crazy formula for cofactor
            m[i].append((
                matrix[(i - 1) % 3][(j - 1) % 3] * matrix[(i + 1) % 3][(j + 1) % 3] - 
                matrix[(i - 1) % 3][(j + 1) % 3] * matrix[(i + 1) % 3][(j - 1) % 3]
            ))

    return m

def inverse(matrix):
    inverse = transpose(cofactor(a))
    d = det(a)

    for i in range(3):
        for j in range(3):
            inverse[i][j] /= d

    return inverse
    
def solve_eq(a, b):
    return mul_mv(inverse(a), b)

# bonus answer: every line, diagonal or column, if multiplied with it's corresponding line, column 
# from the original matrix, gives the determinant

if __name__ == "__main__":
    a, b = parse("input.txt")

    print(a)
    print(b)
    print()

    sol = solve_cramer(a, b)
    print(sol)
    print(mul_mv(a, sol), b)
    print()

    sol = solve_eq(a, b)
    print(sol)
    print(mul_mv(a, sol), b)
    print()
