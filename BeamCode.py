""" 
Beam Deflection Calculator
Please be nice, this is my first python program and only have ever used MATLAB before

Currently calculates the displacement of of beam when a force is applied vertically
"""

import numpy as np
import matplotlib.pyplot as plt

# pre process
Nele=3      # Number of Elements
Nd=4        # Number of Nodes
E=np.ones((Nele))*200     # Young's Modulus  (kN*mm^2)
I=np.array([1.25*10**5, 1.25*10**5, 4*10**4])   # Moment of Inertia (mm^4)
p=np.zeros((Nele))
BeamType='SimplySupported'
#BeamType='OneFixedOneFree'
#BeamType='SimplySupported'

# Specific Input Matrices
coor=np.array([     # Coordinate Matrix (mm)
    [0,0],
    [150,0],
    [225,0],
    [350,0]
    ])

con=np.array([      # Construction Matrix 
    [0,1],
    [1,2],
    [2,3]
    ])

DoF=np.zeros((Nd,2))     # Degrees of Freedom Matrix for reference later
for i in range(Nd):
    DoF[i,:]=[2*i,2*i+1]

F=np.zeros((2*Nd,1))     # Force Matrix
ForceAppliedNodes={1: -3}   # Matrix containing {node: force} Force in (kN)
for node, force in ForceAppliedNodes.items():   # Sets the Forces based on ForceAppliedNodes
    F[int(DoF[node,0])]=force
    
BC=np.zeros((2*Nd,2))   # Boundary Conditions Matrix
if BeamType == 'SimplySupported':   # Determines type of beam. Will add more later
    print("The Beam is Simply Supported")
    BC[int(DoF[0,0]),0]=1       # Constains Node 0 & Node Nd's x directions. (The y direction does not need to be constained due to the penalty method applied below)
    BC[int(DoF[Nd-1,0]),0]=1
elif BeamType == 'OneFixedOneFree':
    BC[int(DoF[0,0]),0]=1       # Constains Node 0's x & y directions
    BC[int(DoF[0,1]),0]=1
else:
    print("Error in BeamType selection. Please use a supported beam type")

K=np.zeros((2*Nd,2*Nd))     # Global Stiffness Matrix

# Building stiffness matrix
for i in range(Nele):
    le=coor[con[i,1],0]-coor[con[i,0],0]   # Length of each element
    
    coeff=((E[i]*I[i])/le**3)       # Coeff of stiffness matrix for consise code
    
    kel=coeff* np.array([    # Element by element Stiffness matrix 
        [12, 6*le, -12, 6*le],
        [6*le, 4*le**2, -6*le, 2*le**2],
        [-12, -6*le, 12, -6*le],
        [6*le, 2*le**2, -6*le, 4*le**2]
    ])
    
    ind=np.concatenate([DoF[con[i,0],:], DoF[con[i,1],:]])
    ind=ind.astype(int)
    K[np.ix_(ind,ind)] += kel

C=np.max(np.abs(K[:]))*1e4   # Penalty Constant, This gets applied to contrained nodes
for i in range(2*Nd):
    if BC[i,0] == 1:
        K[i,i] += C
        F[i] += C*BC[i,1]

Q=np.linalg.solve(K,F)      # Solving for displacements

# Calculating the Reaction forces
Npe=1000      # Nodes per element for vis. purposes only, Arbitrary set @1000
z=np.linspace(-1,1,Npe)
plt.close('all')
fig = plt.figure(1)
counter=0
x=[]
v=[]
for i in range(Nele):
    for ii in range(Npe):
        x1=coor[con[i,0],0]
        x2=coor[con[i,1],0]
        le=x2-x1
        counter += 1
        
        x_cord=(z[ii]+1)*le/2+x1
        x.append(x_cord)
        
        q = np.array([
            Q[int(DoF[con[i, 0], 0])],    # First node, first DOF
            Q[int(DoF[con[i, 0], 1])],    # First node, second DOF  
            Q[int(DoF[con[i, 1], 0])],    # Second node, first DOF
            Q[int(DoF[con[i, 1], 1])]     # Second node, second DOF])
        ])
        
        h1=0.25*(1-z[ii])**2 * (2+z[ii])
        h2=0.25*(1-z[ii])**2 * (1+z[ii])
        h3=0.25*(1+z[ii])**2 * (2-z[ii])
        h4=0.25*(1+z[ii])**2 * (-1+z[ii])
        H=np.array([h1, h2*le/2, h3, h4*le/2])
       
        v_cord=np.dot(H,q)
        v.append(v_cord)
        
x=np.array(x)
v=np.array(v)

plt.plot(x, v, 'r-')
plt.grid(True)
plt.xlabel('Position (mm)')
plt.ylabel('Deflection (mm)')
plt.title('Beam Deflection')
plt.show()

max_disp_idx = np.argmin(v)  # argmin because displacement is negative (downward)
max_disp_value = v[max_disp_idx].item()
max_disp_location = x[max_disp_idx]
plt.plot(max_disp_location, max_disp_value, 'go', markersize=6, label=f'Maximum deflection: {np.min(v):.6f} mm at x = {x[np.argmin(v)]:.1f} mm')
plt.legend()

print(f"Maximum deflection: {np.min(v):.6f} mm at x = {x[np.argmin(v)]:.1f} mm")