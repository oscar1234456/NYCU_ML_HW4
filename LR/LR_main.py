from LR.data_generate.data_generator import data_generator

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



print("123")