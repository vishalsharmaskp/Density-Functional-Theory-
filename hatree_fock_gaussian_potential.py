import numpy as np
import math as math 
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import eigsh
import numpy as np
import numpy as np


omega = 1.0
sigma = 1.0
A =  1.0
x_0 = 0.5
# N = 2  # number of electrons
# bin_size = 100
# step_size = 2*math.pi/bin_size
step_size = 0.01
# np array of positions from -10 to 10 in steps of 0.1
position = np.arange(-10, 10, step_size)

# position = np.arange(-math.pi, math.pi + step_size, step_size)
print("length of position space = ", len(position))
# print("position (min , max, total_bin_size) = ",  position.min(), position.max(), np.shape(position))

def HO_ground_state(x):
    return (omega/math.pi)**(0.25) * np.exp(-0.5 * omega * x**2)

psi_0_1 = np.array([HO_ground_state(x- x_0) for x in position])
psi_0_2 = np.array([HO_ground_state(x + x_0) for x in position])

# plt.plot(position, psi_0_1, label='psi_0_1')
# plt.plot(position, psi_0_2, label='psi_0_2')
# plt.show()

def normalize_wavefunction(psi):
    """Normalizes a wavefunction psi (which is a list of complex numbers)."""
    A = 0 
    for i in psi:
        A += i * np.conjugate(i) 
    A = A * step_size
    print("A = ", A)
    return psi/np.sqrt(A)

def one_body_harmonic_potential(x):
    return 0.5 * omega**(2) * x**2

harmonic_potential_values = np.array([one_body_harmonic_potential(x) for x in position])

V_HO_potential_matrix = np.diag(harmonic_potential_values)

# Assuming V_HO_potential_matrix is your matrix
# plt.imshow(V_HO_potential_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar(label='Value')
# plt.title('V_HO_potential_matrix')
# plt.show()


# plt.plot(position, harmonic_potential_values, label='harmonic_potential')
# plt.show()

def discrete_second_derivative(func):
    second_derivative = []
    for i in range(len(func)):
        if i == 0:
            derivative = 2 * func[i] - 5 * func[i+1] + 4 * func[i+2] - func[i+3]
        elif i == len(func) - 1:
            derivative = 2 * func[i] - 5 * func[i-1] + 4 * func[i-2] - func[i-3]
        else:
            derivative = func[i-1] - 2 * func[i] + func[i+1]
        second_derivative.append(derivative/(step_size**2))
    return second_derivative


def kinetic_energy(psi):
    # returns the kinetic energy of a wavefunction d^{2}psi/dx^{2} as list
    kinetic_energy = []
    for i in range(len(psi)):
        kinetic_energy.append(-0.5 * discrete_second_derivative(psi)[i])
    return kinetic_energy


def interaction_potential(x_1, x_2):
    return A* np.exp(-(x_1 - x_2)**2/sigma**2)

V_interaction_matrix = np.array([[interaction_potential(i, j) for i in position] for j in position])

# x_1 = np.copy(position)
# x_2 = np.copy(position)

# V_interaction = np.array([[interaction_potential(i, j) for i in x_1] for j in x_2])


# plt.imshow(V, origin='upper')
# plt.colorbar()  
# plt.xlabel('x_1')
# plt.ylabel('x_2')
# plt.show()


def H_O_potential_expectation(psi):
    """
    psi: wavefunction expressed as a list of complex numbers
    Takes a wavefunction psi and returns the expectation value of the harmonic oscillator potential.
    """
    V_HO_potential_matrix = np.diag(harmonic_potential_values)
    # return (np.conjugate(normalize_wavefunction(psi)) @  V_HO_potential_matrix @ normalize_wavefunction(psi)) * step_size
    return np.dot(np.dot(np.conjugate(normalize_wavefunction(psi)), V_HO_potential), normalize_wavefunction(psi)) * step_size    

def K_E_expectation(psi):
    #does not give the correct answer
    # i think we need to put appropriate factor to convert the sum to integral 1/L ??
    return np.dot(np.conjugate(normalize_wavefunction(psi)), kinetic_energy(normalize_wavefunction(psi))) * step_size


# Calculate the expectation value of the kinetic energy 
# and petential energy for the grounds state of Harmonic Oscillator

# x = np.copy(position)  # Define x from -pi to pi
## psi = np.sin(x)  # Define psi as sin(x)
# harmonic_oscilator_gs = np.array([HO_ground_state(x) for x in x])
# plt.plot(x, kinetic_energy(psi), label='kinetic_energy')
# plt.plot(x, np.conjugate(psi), label=' cojugate psi')
# plt.plot(x, harmonic_potential_values, label='harmonic_potential')
# plt.show()

# V_HO_potential = np.diag(harmonic_potential_values)
# HO_expectation = H_O_potential_expectation(harmonic_oscilator_gs)
# KE_expectation = K_E_expectation(harmonic_oscilator_gs)
# print("H_O_potential_expectation:", HO_expectation)
# print("K_E_expectation:", KE_expectation)
# print("Total Energy ", HO_expectation + KE_expectation, "expected value = 0.5 hbar * omega ", 0.5)


def density_matrix(list_orbitals):
    # Initialize density_matrix as a zero array with the appropriate shape
    # density matrix gamma_{x, y} is defined as sum_{i} psi^{*}_{i}(y) * psi_{i}(x) 
    density_matrix = np.zeros((len(position), len(position)))
    for orbital in list_orbitals: 
        psi_complex_conjugate = np.conjugate(orbital)
        psi = orbital
        density_matrix += np.array([[j*i for j in psi_complex_conjugate] for i in psi])
    return density_matrix

# print(np.shape(density_matrix([psi_0_1, psi_0_2])))


# Assuming density_matrix is a function that returns a matrix
density_mat = density_matrix([psi_0_1, psi_0_2])

# plt.imshow(density_mat.real, cmap='hot', interpolation='nearest')
# plt.colorbar(label='Value')
# plt.title('Density Matrix')
# plt.show()

def hatree_potential(list_orbitals):
    """
    Take orbital list and return the Hartree potential as a 2d array with the diagonal 
    representing the potential at each position. 
    """
    # V_hatree(x) = sum_{i} V(x, y) * |psi_{i}(y)|^{2} = \sum_{y} V(x, y) * gamma(y, y)
    V_hatree_1 =  V_interaction_matrix @ np.diag(density_matrix(list_orbitals))
    V_hatree_2 =  np.diag(V_hatree_1)
    return V_hatree_2

hatree_potential_value = hatree_potential([psi_0_1, psi_0_2])
# print( "hartree potential = \n", hatree_potential_value)
# print("shape of hartree potential = ", np.shape(hatree_potential_value))


def exchange_correlation_potential(list_orbitals):
    # V_xc(x) = sum_{i} V(x, y) * psi^{*}_{i}(y) * psi_{i}(x) = V(x, y) * gamma(x, y)
    V_xc =  V_interaction_matrix @ density_matrix(list_orbitals)
    return V_xc

# print("Exchange potential = ", exchange_correlation_potential([psi_0_1, psi_0_2]))
# print("shape of exchange potential = ", np.shape(exchange_correlation_potential([psi_0_1, psi_0_2])))



# Get the exchange correlation potential
exchange_potential_value = exchange_correlation_potential([psi_0_1, psi_0_2])

# Create a new figure
# plt.figure()

# Plot the exchange correlation potential
# plt.imshow(hatree_potential_value, cmap='hot', interpolation='nearest')
# plt.colorbar(label='Value')
#plt.settitle('Hatree Potential')
# plt.show()

# plt.imshow(exchange_potential_value, cmap='hot', interpolation='nearest')
# plt.colorbar(label='Value')
# plt.title('Exchange Potential')
# plt.show()

# plt.imshow(hatree_potential_value + exchange_potential_value, cmap='hot', interpolation='nearest')
# plt.colorbar(label='Value')
# plt.title('Total Hatree-fock Potential without the single particle potentials')
# plt.show()



# Create the kinetic energy operator matrix
T = -1 / (2 * step_size**2) * (np.diag(-2 * np.ones(len(position))) + np.diag(np.ones(len(position)-1), k=-1) + np.diag(np.ones((len(position))-1), k=1))

# Modify the first row for forward third-order derivative
# T[0, 0] = -1 / (2 * step_size**2) * 2
# T[0, 1] = -1 / (2 * step_size**2) * (-5)
# T[0, 2] = -1 / (2 * step_size**2) * 4
# T[0, 3] = -1 / (2 * step_size**2) * (-1)

# Modify the last row for backward second derivative
# T[-1, -1] = -1 / (2 * step_size**2) * (2)
# T[-1, -2] = -1 / (2 * step_size**2) * -5
# T[-1, -3] = -1 / (2 * step_size**2) * 4
# T[-1, -4] = -1 / (2 * step_size**2) * -1

# print("T = ", T)
# print("shape of T = ", np.shape(T))

# Create the potential energy operator matrix
V = V_HO_potential_matrix  
# print("V = \n  ", V)
# print("shape of V = ", np.shape(V))


# def Hatree_fock_procedue(orbitals):

#     # calculate the density matrix
#     density_mat = density_matrix(orbitals)
#     # calculate the hatree potential
#     V_hatree = hatree_potential(orbitals)
#     # calculate the exchange potential
#     V_xc = exchange_correlation_potential(orbitals)
#     # return the sum of the hatree and exchange potentials

#     #solving the scrhodinger equation to obtain two lowest lying orbitals
#     H = T + V + V_hatree + V_xc

#     # Find the two smallest eigenvalues and their corresponding eigenvectors
#     eigenvalues, eigenvectors = eigsh(H, 2, which='SM')

#     # The two smallest eigenvalues
#     e_1, e_2 = eigenvalues

#     # Save each eigenvector as a 1D numpy array
#     orbital_1 = np.array(eigenvectors[:, 0])
#     orbital_2 = np.array(eigenvectors[:, 1])

#     # Put the eigenvectors into another 1D numpy array
#     new_orbitals = np.array([orbital_1, orbital_2])

#     # Calculate the total energy
#     total_energy = e_1 + e_2

#     # Print the eigenvalues and total energy
#     print(f"e_1: {e_1}, e_2: {e_2}, total_energy: {total_energy}")

def plot_orbitals(orbitals):
    # Plot the first orbital
    plt.plot(position, orbitals[0], label='Orbital 1')

    # Plot the second orbital
    plt.plot(position , orbitals[1],  label='Orbital 2')

    # Add labels and title
    plt.xlabel('Position')
    plt.ylabel('Wavefunction')
    plt.title('Orbitals')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


nmax = 5

def Hatree_fock_procedue(orbitals):
    run = 0
    for run in range(nmax):
        print("run = ", run)
        # calculate the density matrix
        density_mat = density_matrix(orbitals)
        # calculate the hatree potential
        V_hatree = hatree_potential(orbitals)
        # calculate the exchange potential
        V_xc = exchange_correlation_potential(orbitals)
        # return the sum of the hatree and exchange potentials

        #solving the scrhodinger equation to obtain two lowest lying orbitals
        # H = T + V + V_hatree - V_xc
        # H = T + V 
        H = T + V_hatree - V_xc
        # Find the two smallest eigenvalues and their corresponding eigenvectors
        eigenvalues, eigenvectors = eigsh(H, 2, which='SM')

        # The two smallest eigenvalues
        e_1, e_2 = eigenvalues

        # Save each eigenvector as a 1D numpy array
        orbital_1 = np.array(eigenvectors[:, 0])
        orbital_2 = np.array(eigenvectors[:, 1])

        # Put the eigenvectors into another 1D numpy array
        new_orbitals = np.array([orbital_1, orbital_2])
        orbitals = new_orbitals
    

    # Calculate the total energy
    total_energy = e_1 + e_2

    # Print the eigenvalues and total energy
    print(f"e_1: {e_1}, e_2: {e_2}, total_energy: {total_energy}")
    # print(f"Final orbitals: {new_orbitals}")

    return new_orbitals


initial_eigenvalues, initial_eigenvectors = eigsh(T + V, 2, which='SM')

# The two smallest eigenvalues
e_1, e_2 = initial_eigenvalues

# Save each eigenvector as a 1D numpy array
orbital_1 = np.array(initial_eigenvectors[:, 0])
orbital_2 = np.array(initial_eigenvectors[:, 1])

intitial_orbitals = np.array([orbital_1, orbital_2])
plot_orbitals(intitial_orbitals)
plt.show()
# Run the Hatree-Fock procedure
final_orbitals = Hatree_fock_procedue(intitial_orbitals)

# import matplotlib.pyplot as plt

# Call the function with the obtained orbitals
plot_orbitals(final_orbitals)


