while(iterations < max_iterations and not done):
    # Do Gradient Descent
    ...
    iterations = iterations + 1
    if iterations >= max_iterations:
        break
    # If the change in value for new thetas is too small, we can stop iterating
    done = True
    for k in range(len(thetas)):
        done = abs(thetas[k] - new_thetas[k]) < stopCondition and done
    if done:
        break
