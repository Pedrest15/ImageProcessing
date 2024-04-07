################ HEADER ################
# NAME: Pedro Lucas Castro de Andrade  #
# USP NUMBER: 11212289                 #
# COURSE CODE: SCC0251                 #
# YEAR/SEMESTER: 2024/1                #
# ASSIGNMENT 0 - How to do assignments #
########################################

import imageio.v3 as imageio

if __name__ == '__main__':
    imginput = input().rstrip()
    i = int(input())
    j = int(input())

    img = imageio.imread(imginput)
    r = img[i,j][0]
    g = img[i,j][1]
    b = img[i,j][2]
    print(f"{r} {g} {b}")