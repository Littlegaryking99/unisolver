How to add objective function
=============================

Objective function of a QP problem is the performance standard of the system. 
And the value of variables which lead to the best performance of objective 
function is the solution of QP problem.

In this section we will show how to add objective
function to the QP problem. For linear programming
problems, the objective function is linear, while for  
QP, the objective function is in quadratic form.

In general, we use **+=** to add objective function::

    import unisolver
    prob = unisolver.QpProblem("myproblem", "quadprog")
    prob += x ** 2 * 2 + y ** 2 * 5 + x + y * 5
    prob.objective
    #x + 2*x**2 + 5*y + 5*y**2
    prob.P
    #[[2 0]
    # [0 5]]
    prob.q
    #[[1]
    # [5]]


And it is also supported to add objective function in standard form::

    import unisolver
    import numpy as np
    prob = unisolver.QpProblem("myproblem", "quadprog")
    P = np.array((2, 0, 0, 2)).reshape(2,2)
    q = np.array((1, 5)).reshape(2,1)
    c = QpMVariable("c", [2,1], 0, 3, value = 2)
    prob += c.T.dot(P).dot(c) + q.T.dot(c)

The right part of **+=** will be recognized as 

a *QpExpression* object. It can transfer into list of dictionary format::

    import unisolver
    prob = unisolver.QpProblem("myproblem", "quadprog")
    prob += x ** 2 * 2 + y ** 2 * 5 + x + y * 5
    prob.objective.toDict()
    #[{'name': 'x**2', 'value': 2}, {'name': 'y**2', 'value': 5}, {'name': 'x', 'value': 1}, {'name': 'y', 'value': 5}]

If there exists objective function, they can directly add to original
objective function by **+=**::

    import unisolver
    prob = unisolver.QpProblem("myproblem", "quadprog")
    prob += x ** 2 * 2 + y ** 2 * 5 + x + y * 5
    prob.objective
    #x + 2*x**2 + 5*y + 5*y**2
    prob += x * 2
    #3*x + 2*x**2 + 5*y + 5*y**2

If the objective function not in linear or quadratic form, it will return warning message::

    import unisolver
    prob = unisolver.QpProblem("myproblem", "quadprog")
    prob += x ** 3 * 2 + y ** 2 * 5 + x + y * 5
    #This is not a valid objective function for unisolver

After user input correct objective function, they can add contraints and finally solve the model.
Please refer to next part for further information.
