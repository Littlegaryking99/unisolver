Specify variables for QP problems
==================================

In this section, we will show how to 
specify variables for the model.

After we initialize the QP problem and declare 
the solver that used to solve the QP, we can 
add variables to the problem. 

There are many ways to define the variables. 
The simplest way is directly specify *QpVariable* 
object::

    import unisolver
    x = QpVariable("x", 0, 3)

User can also call a list of *QpVariable* objects by loop definition::

    import unisolver
    x_name = ['x_0', 'x_1', 'x_2']
    x = [QpVariable(x_name[i], 0, 10) for i in range(3)]

User can also a *QpMVariable* to input a matrix of QpVarible by specifying size, name and bounds::

    import unisolver
    b = QpMVariable("b", [2,2], 0, 3, value = 1)
    b = np.asarray(b)            

The structure of *QpMVariable* is initialized with a numpy array, which supports some numpy functions, including *tr()*, *dot* ::

    import unisolver
    c = QpMVariable("c", [2,2], 0, 3, value = 2)
    c1 = np.asarray(c)
    d = np.array((2,2)).reshape(2,1)
    c1.dot(d)
    #[[2*c00 + 2*c01 + 0]
    # [2*c10 + 2*c11 + 0]]

If the name of problem is not specified, it will 
automatically named by **NoName**::

    import unisolver
    x = QpVariable(lowbound = 0, upbound = 3)
    x.name
    #NoName

If lowbound or upbound of the variable is not specified, it 
would be initialized by None. And the initial
value of the variable is set to be 0 if not 
initialized::

    import unisolver 
    x = QpVariable("x")
    x.lowbound
    #None
    x.upbound
    #None
    x.value
    #0

If user made some mistakes when specifying parameters of *QpVariables*, they can fix them 
by *fixlowbound* and *fixupbound* function::

    import unisolver
    x = QpVariable("x", 1)
    x.lowbound
    #1
    x.fixlowbound(2)
    x.lowbound
    #2
    x.upbound
    #None
    x.fixupbound(3)
    x.upbound
    #3

After user create the variables of the model, they can set objective function and constraints for 
the model. Please refer to next part for further information.
