if len(costs) > 0 and cost > costs[-1]:
    learningRate *= alphaFactor
    if retryCount < retryMax:
        retryCount += 1
    else:
        done = true
