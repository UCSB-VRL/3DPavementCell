import numpy as np

def regionGrowing(grayImg,seed,threshold):
    [maxZ, maxY,maxX] = grayImg.shape

    pointQueue = []
    pointQueue.append((seed[0], seed[1],seed[2]))
    outImg = np.zeros(grayImg.shape)
    outImg[seed[0], seed[1],seed[2]] = 1

    pointsNum = 1
    pointsMean = float(grayImg[seed[0], seed[1],seed[2]])


    Next26 = [[-1, -1, -1],[-1, 0, -1],[-1, 1, -1],
                [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
                [-1, 0, 1], [-1, 0, 0],[-1, 0, -1],
                [0, -1, -1], [0, 0, -1], [0, 1, -1],
                [0, 1, 0],[-1, 0, -1],
                [0, -1, 0],[0, -1, 1],[-1, 0, -1],
                [0, 0, 1],[1, 1, 1],[1, 1, -1],
                [1, 1, 0],[1, 0, 1],[1, 0, -1],
                [1, -1, 0],[1, 0, 0],[1, -1, -1]]

    while(len(pointQueue)>0):

        growSeed = pointQueue[0]
        del pointQueue[0]

        for differ in Next26:
            growPointz = growSeed[0] + differ[0]
            growPointy = growSeed[1] + differ[1]
            growPointx = growSeed[2] + differ[2]

           
            if((growPointx < 0) or (growPointx > maxX - 1) or
               (growPointy < 0) or (growPointy > maxY - 1) or (growPointz < 0) or (growPointz > maxZ - 1)) :
                continue

        
            if(outImg[growPointz,growPointy,growPointx] == 1):
                continue

            data = grayImg[growPointz,growPointy,growPointx]
            
            if(abs(data - pointsMean)<threshold):
                pointsNum += 1
                pointsMean = (pointsMean * (pointsNum - 1) + data) / pointsNum
                outImg[growPointz, growPointy,growPointx] = 1
                pointQueue.append([growPointz, growPointy,growPointx])
        #print np.amax(-outImg)
    return outImg
