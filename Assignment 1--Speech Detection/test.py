global_variable = 10  # 这是一个全局变量

def modify_global_variable(global_variable):
    global_variable = 20  # 修改全局变量的值

modify_global_variable(global_variable)  # 调用函数来修改全局变量的值
print(global_variable)  # 输出修改后的全局变量的值，输出为 20
