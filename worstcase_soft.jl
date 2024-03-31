#= Code to check non-identifiability for soft interventions.
We define two sets of model parameters and set the respective products we have access to from \kappa_3(X) equal to each other.
To test identifiability we examine the solution set of the resulting system of equations. =#

# Load required packages
using Oscar, LinearAlgebra

# Define polynomial ring with variables of interest
R, l, lt, l1, l1t, d, dt, dint, dtint, h, ht = polynomial_ring(QQ, "l" => (1:2,1:2), "lt" => (1:2,1:2), "l1" => (1:2,1:2), "l1t" => (1:2,1:2), "d" => 1:2, "dt" => 1:2, "dint" => 1:2, "dtint" => 1:2, "h" => (1:2,1:2), "ht" => (1:2,1:2))

# Define matrices of unknowns
H = h
Ht = ht
L = [1 -l[1,2]; 0 1]
Lt = [1 -lt[1,2]; 0 1]

L1 = [1 -l1[1,2]; 0 1]
L1t = [1 -l1t[1,2]; 0 1]

D = [d[1] 0; 0 d[2]] 
Dt = [dt[1] 0; 0 dt[2]]

D1 = [dint[1] 0; 0 d[2]]
D1t = [dtint[1] 0; 0 dt[2]]

D2 = [d[1] 0; 0 dint[2]]
D2t = [dt[1] 0; 0 dtint[2]]

C = D*L*H
Ct = Dt*Lt*Ht
C1 = D1*L1*H
C1t = D1t*L1t*Ht
C2 = D2*L*H
C2t = D2t*Lt*Ht

eqnsA = C-Ct
eqnsA1 = C1-C1t
eqnsA2 = C2-C2t

# Create ideals
I = ideal(R, [reduce(vcat, eqnsA); reduce(vcat, eqnsA1); reduce(vcat, eqnsA2);
                  sum(H[1,:].^2) - 1; sum(Ht[1,:].^2) - 1; sum(H[2,:].^2) - 1; sum(Ht[2,:].^2) - 1;])
J = ideal(R, [reduce(vcat, H-Ht); reduce(vcat, l-lt); reduce(vcat, l1-l1t); reduce(vcat, d-dt); reduce(vcat, dint-dtint); h[1,1]*h[2,2]-h[1,2]*h[2,1]])

# Impose that the parameters are different across models and that the norms of the rows of H equal 1
K = saturation(I,J)

# Determine dimension of solution set
ind_cond = codim(K) 
difference = 20-ind_cond
println("The system of equations has $ind_cond independent conditions and 20 variables, so the solution set is $difference-dimensional.")