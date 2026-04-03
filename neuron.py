def step (x):
    if x > 0:
        return 1
    else:
        return 0
    
def perceptron(x1,x2,w1,w2,b):
    sum = x1*w1 + x2*w2 + b
    output = step(sum)
    return output       
print(perceptron(1,1,0.5,0.5,-0.7))        
print(perceptron(1,0,0.5,0.5,-0.7))
print(perceptron(0,1,0.5,0.5,-0.7))
print(perceptron(0,0,0.5,0.5,-0.7))