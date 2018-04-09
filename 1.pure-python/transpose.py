#implementation of transpose on pure python
test2 = [[1, -2, 1],
         [0, 2, -8],
         [-4, 5, 9],
         [5, 6, 7]]

test1 = [[1, 1, 1, 1, 1, 2, 3]]

test3 = [[1, 1, -2, 1, 3, -1],
         [2, -1, 1, 2, 1, -3],
         [1, 3, -3, -1, 2, 1],
         [5, 2, -1, -1, 2, 1],
         [-3, -1, 2, 3, 1, 3],
         [4, 3, 1, -6, -3, -2]]

def transpose(mat):
    list_ = [[x for x in range(len(mat))] for _ in range(len(mat[0]))]
    for i in range(len(list_)):
        for j in range(len(list_[0])):
            list_[i][j] = mat[j][i]
    return list_

print(transpose(test3))
