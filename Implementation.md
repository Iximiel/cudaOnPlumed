# Implementation

The coordination is more or less calculated with 
$\frac{1}{2} \sum^{i=0}_{i<nat}\sum^{j=0}_{j<nat,j\neq i}f(d_{ij})$
, where $d_{ij}$ is the distances between atom $i$ and $j$, and $f(x)$ is a function that usually has the form

$$
f(x)=\left\{ \begin{array}{lr}
1 & x\leq d_0\\
s(x) &d_0<x\leq d_{max}\\
0 & x > d_{max} 
\end{array}\right.
$$

and $s(x)$ is a switching function links smoothly 1 to 0 within $d_0$ and $d_{max}$

In this case I used the RATIONAL function like in the default plumed implementation: $s(r)=\frac{ 1 - \left(\frac{ r - d_0 }{ r_0 }\right)^{n} }{ 1 - \left(\frac{ r - d_0 }{ r_0 }\right)^{m} }$.

The implementation of the rational function is linear, and follows the one in plumed. But I have not implemented with d_0, so $s(r)=\frac{ 1 - \left(\frac{ r }{ r_0 }\right)^{n} }{ 1 - \left(\frac{ r }{ r_0 }\right)^{m} }$ plus a bit of stretc, because $s(d_{max}) = shift $ : $s^s(x)=s(x)*stretch+shift$

