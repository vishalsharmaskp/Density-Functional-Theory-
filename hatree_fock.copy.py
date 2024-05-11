import numpy as np
import math as math 
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import eigsh
import numpy as np
import numpy as np
import sys

print("float point info= ", sys.float_info)

omega = 0.5 
sigma = 0.5
A =  1.0
x_0 = 0.5
lmbda = 0.0
step_size = 0.25
position = np.arange(-50, 50, step_size)

print("length of position space = ", len(position))
print("position (min , max, total_bin_size) = ",  position.min(), position.max(), np.shape(position))

def HO_ground_state(x):
    return (omega/math.pi)**(0.25) * np.exp(-0.5 * omega * x**2)

def gaussian_wavepacket(x, x_0, sigma):
    return (1/(2*np.pi*sigma**2))**(1/4) * np.exp(-(x - x_0)**2/(4*sigma**2))

psi_0_Gaussian = np.array([gaussian_wavepacket(x, x_0, sigma) for x in position])
psi_1_Gaussian = np.array([gaussian_wavepacket(x, -x_0, sigma) for x in position])

psi_0_1 = np.array([HO_ground_state(x- x_0) for x in position])
psi_0_2 = np.array([HO_ground_state(x + x_0) for x in position])



def normalize_wavefunction(psi):
    """Normalizes a wavefunction psi (which is a list of complex numbers)."""
    A = 0 
    for i in psi:
        A += i * np.conjugate(i) 
    A = A * step_size
    return psi/np.sqrt(A)


def plot_orbitals(orbitals):
    # Plot the first orbital
    n = 0 
    for i in orbitals:
        normalize_orbital = normalize_wavefunction(i)
        plt.plot(position, normalize_orbital, label='Orbital' + str(n))
        # Add labels and title
        plt.xlabel('Position')
        plt.ylabel('Wavefunction')
        plt.title('Orbitals')
        # plt.savefig(f"attractive_hatreeonly_orbital.png")
        # Add legend
        plt.legend()
        n += 1
    # Show the plot
    plt.show()

def one_body_harmonic_potential(x):
    """
    the parameter
    """
    return lmbda * 0.5 * omega**(2) * x**2

harmonic_potential_values = np.array([one_body_harmonic_potential(x) for x in position])

V_HO_potential_matrix = np.diag(harmonic_potential_values)

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


def K_E_expectation(psi):
    #does not give the correct answer
    return np.dot(np.conjugate(normalize_wavefunction(psi)), \
                   kinetic_energy(normalize_wavefunction(psi))) * step_size

def H_O_potential_expectation(psi):
    """
    psi: wavefunction expressed as a list of complex numbers
    Takes a wavefunction psi and returns the expectation value of the harmonic oscillator potential.
    """
    V_HO_potential_matrix = np.diag(harmonic_potential_values)
    return (np.conjugate(normalize_wavefunction(psi)) @ \
             V_HO_potential_matrix @ normalize_wavefunction(psi)) * step_size
    # return np.dot(np.dot(np.conjugate(normalize_wavefunction(psi)), \
                        #  V_HO_potential_matrix), normalize_wavefunction(psi)) * step_size  

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


# Create the kinetic energy operator matrix
T = -1 / (2 * step_size**2) * (np.diag(-2 * np.ones(len(position))) \
                               + np.diag(np.ones(len(position)-1), k=-1) + np.diag(np.ones((len(position))-1), k=1))

# potential energy operator matrix
V = V_HO_potential_matrix

H_harmmonic = T + V

# # Find the two smallest eigenvalues and their corresponding eigenvectors
# HO_eigenvalue, HO_eigenvectors =  np.linalg.eigh(H_harmmonic)
# # The two smallest eigenvalues  of the harmonic oscillator 
# HO_e_1, HO_e_2 = HO_eigenvalue[0: 2]

# print("Harmonic oscillator eigenvalues = ", HO_eigenvalue[0: 5])

# # eigenvector corresponding to two lowest eigenvalues of harmonic oscillator 
# orbital_1_HO = np.array(HO_eigenvectors[:, 0])
# orbital_2_HO = np.array(HO_eigenvectors[:, 1]) 
# orbital_3_HO = np.array(HO_eigenvectors[:, 2])
# orbital_4_HO = np.array(HO_eigenvectors[:, 3])
# HO_eigenvectors = np.array([orbital_1_HO, orbital_2_HO, orbital_3_HO, orbital_4_HO])
# plot_orbitals(HO_eigenvectors)
# print("H_O_potential_expectations:", H_O_potential_expectation(orbital_1_HO), H_O_potential_expectation(orbital_2_HO),
#                             H_O_potential_expectation(orbital_3_HO), H_O_potential_expectation(orbital_4_HO))
# print("K_E_expectations:", K_E_expectation(orbital_1_HO), K_E_expectation(orbital_2_HO), K_E_expectation(orbital_3_HO)
#                                            , K_E_expectation(orbital_4_HO))
# plt.plot(position, harmonic_potential_values)
# plt.show()


#####################################################################################################################
#putting the interaction into the code



# Define the interaction potential
def interaction_potential(x_1, x_2):
    return A* np.exp(-(x_1 - x_2)**2/sigma**2)

# create the interaction matrix V(x, y) = A * exp(-(x - y)^{2}/sigma^{2}) where x and y are the position space
V_interaction_matrix = np.array([[interaction_potential(i, j) for i in position] for j in position])


#create a "density matrix" gamma_{x, y} = sum_{i} psi^{*}_{i}(y) * psi_{i}(x)
def density_matrix(list_orbitals):
    # Initialize density_matrix as a zero array with the appropriate shape
    # density matrix gamma_{x, y} is defined as sum_{i} psi^{*}_{i}(y) * psi_{i}(x) 
    density_matrix = np.zeros((len(position), len(position)))
    for orbital in list_orbitals: 
        psi_complex_conjugate = np.conjugate(orbital)
        psi = orbital
        density_matrix += np.array([[j*i for j in psi_complex_conjugate] for i in psi])
    return density_matrix

def hatree_potential(list_orbitals):
    """
    Take orbital list and return the Hartree potential as a 2d array with the diagonal 
    representing the potential at each position and off-daiagonal being zero.
    """
    # V_hatree(x) = sum_{i} V(x, y) * |psi_{i}(y)|^{2} = \sum_{y} V(x, y) * gamma(y, y)
    V_hatree_1 =  V_interaction_matrix @ np.diag(density_matrix(list_orbitals))
    V_hatree_2 =  np.diag(V_hatree_1)
    return V_hatree_2

# print("shape of hatree potential = ", np.shape(hatree_potential(orbitals)))

def exchange_correlation_potential(list_orbitals):
    """
    Takes a 2d array of density matrix and returns the exchange correlation potential as a 2d array.
    """
    # V_xc(x) = sum_{i} V(x, y) * psi^{*}_{i}(y) * psi_{i}(x) = V(x, y) * gamma(x, y)
    V_xc =  V_interaction_matrix @ density_matrix(list_orbitals)
    return V_xc

# print("shape of exchange potential = ", np.shape(exchange_correlation_potential(orbitals)))


# setting up the hatree focl procedure
 
# number of iteration
nmax = 500

def Hatree_fock_procedue(orbitals):
    """Need to give a list of orbitals as input and return the final orbitals after nmax iterations."""
    orbital_1 = orbitals[0]
    orbital_2 = orbitals[1]
    total_density_old = np.abs(orbital_1)**2 + np.abs(orbital_2)**2
    print("old density shape = ",  np.shape(total_density_old))
    plt.plot(position, total_density_old, label='total density old')
    plt.legend()
    plt.show()

    has_converged = False

    iteration = 0
    for run in range(nmax):

        print("iteration  = ", iteration)

        # calculate the density matrix
        density_mat = density_matrix(orbitals)

        # calculate the hatree potential
        V_hatree = hatree_potential(orbitals)

        # calculate the exchange potential
        # V_xc = exchange_correlation_potential(orbitals)
        # return the sum of the hatree and exchange potentials

        #solving the scrhodinger equation to obtain two lowest lying orbitals
        # H = T + V + V_hatree - V_xc
        H_hatree = T + V + V_hatree

        # Find the two smallest eigenvalues and their corresponding eigenvectors
        # eigenvalues, eigenvectors = eigsh(H, 2, which='SM', maxiter=10000) 
        eigenvalues, eigenvectors = np.linalg.eigh(H_hatree)

        # The two smallest eigenvalues
        e_1, e_2 = eigenvalues[0: 2]

        # Save each eigenvector as a 1D numpy array
        orbital_1 = np.array(eigenvectors[:, 0])
        orbital_2 = np.array(eigenvectors[:, 1])
            
        probability_density_1 = np.conjugate(orbital_1) * orbital_1
        probability_density_2 = np.conjugate(orbital_2) * orbital_2
        total_density_new = probability_density_1 + probability_density_2

        iteration += 1

        if np.allclose(total_density_new, total_density_old, atol=1e-2):
            has_converged = True
            print("Converged after", run, "iterations")
        
        if has_converged:
            plt.plot(position, orbital_1, label='orbital_1')
            plt.plot(position, orbital_2, label='orbital_2')
            plt.show()

            plt.plot(position, total_density_new, label='total density new')
            plt.legend()
            plt.show()

            total_energy = e_1 + e_2
            print(f"e_1: {e_1}, e_2: {e_2}, total_energy: {total_energy}")
            break 
        
        if iteration % 51 == 0:
            plt.plot(position, total_density_new, label='total density new')
            plt.legend()
            plt.show() 
        # Put the eigenvectors into another 1D numpy array
        orbitals = np.array([orbital_1, orbital_2])
     
        total_density_old = total_density_new

    print("did not converge after ", nmax, "iterations")
    plt.plot(position, orbital_1, label='orbital_1')
    plt.plot(position, orbital_2, label='orbital_2')
    plt.show()

    plt.plot(position, total_density_new, label='total density new')
    plt.legend()
    plt.show()

    total_energy = e_1 + e_2
    print(f"e_1: {e_1}, e_2: {e_2}, total_energy: {total_energy}")
    return None

initial_eigenvalues, initial_eigenvectors = eigsh(T + V, 2, which='SM')

# The two smallest eigenvalues
e_1, e_2 = initial_eigenvalues

# Save each eigenvector as a 1D numpy array
# orbital_1 = np.array(initial_eigenvectors[:, 0])
# orbital_2 = np.array(initial_eigenvectors[:, 1])

orbital_1 = np.array(psi_0_1)
orbital_2 = np.array(psi_0_2)

intitial_orbitals = np.array([orbital_1, orbital_2])



plot_orbitals(intitial_orbitals)

#Run the Hatree-Fock procedure
final_orbitals = Hatree_fock_procedue(intitial_orbitals)

