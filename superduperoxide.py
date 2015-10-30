################################################################################
# superduperoxide.py
#
# Simulates discharge curves of a Li/O2 battery by modeling lithium peroxide
# nucleation and growth with a transient lithium superoxide concentration profile.
#
# Author: Sampson Lau (sl2427@cornell.edu)
# Affiliation: Archer Research Group
#              School of Chemical and Biomolecular Engineering
#              Cornell University
################################################################################

## Import necessary packages
import numpy as np
import math
## Physical constants
from scipy import constants as const

##################################################
# Define parameters
##################################################

# Experiment parameters
I = 0.2*1e-3              # Discharge current [A]
T = 294.35                # Temperature [K]
# Cathode parameters
active_mass = 2e-6*0.9    # Mass of active material on cathode [kg]
area_geo = const.pi*(5.0/8/2 * 0.0254)**2 # Geometric area of cathode [m^2]
specific_area = 62*1000   # Super P - Surface area per gram of carbon [m^2/kg]

## Superoxide diffusion model parameters
# Approximate superoxide diffusivity as O2 diffusivity in TEGDME = 2.17e-6 cm^2/s
# source: #Laoire2010
D = 2.17e-10 			# [m^2/s]
# Superoxide dismutation rate constant [m/s]
k_d = 2.9e-9
# Nucleation rate constant [m/s*mol]
k_n = 4.5729251e+17*const.e*k_d*const.N_A
# Characteristic length [m]
L = 11.9e-9

# Empirical parameters
N_0 = 2.93962e12                      # Number of initial nuclei [#/m^2]
alpha = 0.656                         # Charge transfer coefficient of superoxide formation
i_0_geo = 0.003717533                 # Exchange current density (geometric) [A/m^2]
E_0 = 2.8531                          # Equilibrium cell voltage [V]
resis = 0.062443785                   # Bulk resistivity [ohm*m^2]

## These numbers don't change as long as you're in this universe!
# Physical properties of lithium peroxide
density = 2310                        # Density of lithium peroxide [kg/m^3]
MW = 45.881                           # Molecular weight of lithium peroxide [g/mol]
# Fundamental constants
F = const.e*const.N_A                 # Faraday constant [C/mol]

# Intermediate calculations
# Express constants in more convenient forms
area_real = active_mass*specific_area # Real surface area [m^2]
roughness = area_real/area_geo        # Ratio of real:geometric surface area [m^2]
i_geo = I/area_geo                    # Geometric current density [A/m^2]
i = I/area_real                       # Real current density [A/m^2]
i_0 = i_0_geo/roughness               # Exchange current density (real) [A/m^2]
V_rate = i*MW/(2*F*1000*density)      # Volumetric growth rate per area [m^3/(m^2*s)]

# Characteristic scaling factors for particle growth
#r_c = (2*const.pi*N_0)**(-0.5)        # Characteristic particle radius [m]
r_g = (const.pi*N_0)**(-0.5)          # Characteristic particle radius [m]
t_g = (2*const.pi*N_0*r_g**3)/V_rate  # Characteristic time of growth [s]

# Characteristic scaling factors for diffusion
t_d = L**2/D            # The characteristic diffusion time [s]
Da = k_d*L/D              # The Damkholer number
c_ss = i/(k_d*F)          # Saturation concentration of superoxide [mol/m^3]

# Define timesteps
_dt_g = 5e-5             # Growth time step [non-dimensional]
_tsat = 7/Da             # Time it takes for concentration to saturate [non-dimensional]
_dt_d = _tsat/1e2        # Diffusion time step [non-dimensional]
# Define gridpoints to use for finite differences method of superoxide diffusion
gridpts = 51            # Number of points
dX = 1.0/(gridpts-1)	# Grid spacing
# Define (relative) tolerance of the radius-seeking algorithm
tolerance = 1e-10

# Define time step sizes for each phase of the model
# Phase 1: Development of the LiO2 concentration profile
if _dt_d*t_d < _dt_g*t_g:
	dt_1 = _dt_d*t_d
else:
	dt_1 = _dt_g*t_g
# Phase 2: LiO2 Concentration profile nears saturation
if (3*_dt_d*t_d) < _dt_g*t_g:
	dt_2 = 3*_dt_d*t_d
else:
	dt_2 = _dt_g*t_g
# Phase 3: LiO2 Concentration is saturated
dt_3 = _dt_g*t_g

##################################################
# Functions
##################################################

# Overpotential
def overpotential(theta):
	"Determines the overpotential as a function of cathode surface coverage."
	eta = const.k*T/(const.e*alpha)*np.log(i/i_0*1/(1-theta))
	return eta

# The rate of nucleation
def nuclerate(c_surf):
	"Determines the nucleation rate as a function of overpotential and coverage."
	J = k_n*c_surf
	return J

# The growth function - should equal zero when solved for the correct radius
def gfunc(r_guess,theta,_N,r0,c_surf,_dt):
	"Solve for r_guess to determine particle radius at next step."
	j = len(r0)-1
	f = (1-theta[j])*sum(_N[0:j+1]*(r_guess**3/3 - r_guess**2*r0[0:j+1] \
								   + r_guess*r0[0:j+1]**2 - r0[j]**3/3 \
								   + r0[j]**2*r0[0:j+1] - r0[j]*r0[0:j+1]**2)) \
		- _dt*c_surf/c_ss
	return f

# The derivative of the above growth function
def gderiv(r_guess,theta,_N,r0):
	"Derivative of above function; used for Newton-Raphson method."
	j = len(r0)-1
	f_prime = (1-theta[j])*sum(_N[0:j+1]*(r_guess**2 - 2*r_guess*r0[0:j+1] \
	                                	 + r0[0:j+1]**2))
	return f_prime

# Grow to the next time step
def grow(theta,_N,r0,c_surf,_dt):
	"Iteratively solves for next radius using Newton-Raphson method."
	j = len(r0)-1
	thisguess = r0[j]
	nextguess = thisguess - gfunc(thisguess,theta,_N,np.array(r0),c_surf,_dt) \
	                        /gderiv(thisguess,theta,_N,np.array(r0))
	while 2*abs(thisguess - nextguess)/(thisguess + nextguess) > tolerance:
		thisguess = nextguess
		nextguess = thisguess - gfunc(thisguess,theta,_N,np.array(r0),c_surf,_dt) \
		                        /gderiv(thisguess,theta,_N,np.array(r0))
	return nextguess

# Coverage
def coverage(_N,r0):
	"Calculates the coverage from the sum of particle radii."
	j = len(r0)-1
	theta = 1 - np.exp(-0.5*np.sum(N_0*_N[0:j+1]*(r0[j]-r0[0:j+1])**2))
	return theta

# Solve for the next conecentration profile
def fdm_solve(_c,A,b,_dt):
# _dt is the timestep, and can be varied
	for k in range(1,gridpts-1):
		A[k] = np.zeros(gridpts)
		A[k][k-1] = 1.0
		A[k][k] = -2*dX**2/_dt-2
		A[k][k+1] = 1.0
		b[k] = (-_c[-1][k-1] + (2-2*dX**2/_dt)*_c[-1][k] - _c[-1][k+1])

	## Solve the system of equations
	return np.linalg.solve(A,b)

##################################################
# Initializations
##################################################

# Initalize model variables
t = [0.0]      # The time array [s]
_N = [1]         # Number density of particles, non-dimensional [#/m^2]
r0 = [0.0]       # Radius of particles that nucleated at t = 0 [m]
theta = [0.0]    # The coverage
eta = [overpotential(theta[0])] # The overpotential [V]
V = [E_0 - eta[0]] # The cell voltage [V]

# More initalizations...
_c = [np.zeros(gridpts)]   # Non-dimensional concentration [mol/m^3]
                           # (initial condition: no superoxide anywhere)
c_surf = [0]               # Concentration at the surface [mol/m^3]
# Build system of equations for diffusion model
A = np.zeros((gridpts,gridpts))    # Matrix
b = np.zeros(gridpts)              # Coefficients
## Boundary condition at cathode surface: flux balance with discharge current
A[0][0] = Da*dX+1
A[0][1] = -1.0
b[0] = (Da*dX)
## Boundary condition at characteristic confinement length: no flux
A[gridpts-1][gridpts-2] = -1.0
A[gridpts-1][gridpts-1] = 1.0

##################################################
# Run script
##################################################

## Define the functions to be looped
def concentration_advance(_dt):
	"Calculate concentration profile of next time step."
	_c.append(fdm_solve(_c,A,b,_dt))
	c_surf.append(c_ss*_c[-1][0])

def peroxide_advance(_dt,c_surf):
	"Grow and nucleate lithium peroxide particles to next time step."
	r0.append(grow(theta,_N,r0,c_surf,_dt))
	_N.append(_dt*t_g*nuclerate(c_surf)/N_0)

def theta_eta():
	"Calculate the next timestep's coverage and overpotential."
	theta.append(coverage(np.array(_N),r_g*np.array(r0)))
	eta.append(overpotential(theta[-1]))

def step(dt):
	"Advance simulation to next time step."
	# dt is the size of the next time step
	concentration_advance(dt/t_d)
	peroxide_advance(dt/t_g,c_surf[-1])
	theta_eta()
	t.append(t[-1]+dt)

def step_first(dt):
	"Advance simulation to first time step. (Special case of above function.)"
	# dt is the size of the next time step
	concentration_advance(dt/t_d)
	r0.append((3*dt_1/t_g*(c_surf[-1]/c_ss))**(1./3)) # First growth step is analytically solvable
	_N.append(dt_1*nuclerate(c_surf[-1])/N_0)
	theta_eta()
	t.append(t[-1]+dt)

def step_ss(dt):
	"Advance simulation to next time step, assuming LiO2 concentration is at steady-state."
	# dt is the size of the next time step
	peroxide_advance(dt/t_g,c_ss)
	theta_eta()
	t.append(t[-1]+dt)

step_first(dt_1)                 # Run the model for the first time step
while t[-1] < (4.0/7)*_tsat*t_d: # PHASE 1
	step(dt_1)                   #   Development of LiO2 concentration profile
while t[-1] < _tsat*t_d:         # PHASE 2
	step(dt_2)                   #   Concentration profile approaches steady state
while t[-1] < 36000:             # PHASE 3
	step_ss(dt_3)                #   Concentration profile is at steady state
	if eta[-1] > 2.0:            #   Break out of the loop when a large drop in
		break#dance              #   overpotential is detected.

Q = I*np.array(t)/3.6        # Capacity discharged [mAh]
V = E_0-np.array(eta)        # Voltage at each time step [V]

##################################################
# Plot discharge curve
##################################################

# Import the necessary packages
from matplotlib import pyplot as plt

# Pin end of discharge curve to zero
V[-1] = 0.0

# Create plot
plt.plot(Q,V)
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,2.0,3.0))
plt.xlabel("Capacity [mAh]")
plt.ylabel("Voltage [V]")
plt.show()
