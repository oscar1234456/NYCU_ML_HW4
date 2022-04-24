from LR.data_generate.create import create_phi, create_t
from LR.data_generate.data_generator import data_generator
from LR.process.confusion import confusion_matrix
from LR.process.gradient_descent import gradient_descent
from LR.process.newton import newton_method

config1 = {"N": 50, "mx1": 1, "my1": 1, "mx2": 10,
           "my2": 10, "vx1": 2, "vy1": 2, "vx2": 2, "vy2": 2}
config2 = {"N": 50, "mx1": 1, "my1": 1, "mx2": 3,
           "my2": 3, "vx1": 2, "vy1": 2, "vx2": 4, "vy2": 4}

using_config = {"N": 50, "mx1": 1, "my1": 1, "mx2": 10,
                "my2": 10, "vx1": 2, "vy1": 2, "vx2": 2, "vy2": 2}

# input mode
mode = int(input("mode(0:free/1:Case1/2:Case2)"))
if mode == 0:
    for key in using_config:
        using_config[key] = int(input(f"please input {key}:"))
    print("--------------")
    print("config:")
    print(using_config)
elif mode == 1:
    using_config = config1
    print("--------------")
    print("config:")
    print(using_config)
else:
    using_config = config2
    print("--------------")
    print("config:")
    print(using_config)

# data generate
# data_generator(N, mx, my, vx, vy)
class_1 = data_generator(using_config["N"], using_config["mx1"],
                         using_config["my1"], using_config["vx1"],
                         using_config["vy1"])
class_2 = data_generator(using_config["N"], using_config["mx2"],
                         using_config["my2"], using_config["vx2"],
                         using_config["vy2"])

# w = (3, 1) linear: w1x+w2y+w3 = result => result<0 -> t==0
phi = create_phi(class_1, class_2)  # phi: (2*N, dim(w)=3)
t = create_t(using_config["N"])

# gradient
print("Gradient descent:")
w_gradient = gradient_descent(phi, t)
print("w:")
print(w_gradient)
# TODO: confusion matrix
confusion_matrix(phi, w_gradient, t)
print("-----------------------")

# Newton-Raphson
print("Newton's method:")
w_newton = newton_method(phi, t)
print("w:")
print(w_newton)
confusion_matrix(phi, w_newton, t)

print()
