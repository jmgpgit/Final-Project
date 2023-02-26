# Final-Project

## Network Optimisation

First we create the basic framework on which we can work. We require the use of classes and inheritance to attain the maximal generality in order to fully investigate our problem.

Network architecture:


```mermaid

flowchart TD
    A[Vertex] -->|add weight| B(Weighted Vertex)
    E[Edge] -->|add weight| G[Weighted Edge]
    E[Edge] -->|add direction| H[Directed Edge]
    G[Weighted Edge] --->|add direction| I
    H[Directed Edge] --->|add weight| I
    I(Directed Weighted Edge) <--> C{Network}
    B <--> C{Network}

    C -->|flow| D[Path]
    C -->|flow| X[Path]
    C -->|flow| Z[Path]
    C -->|flow| Y[...]



    D -->J[Optimization]
    Z -->J[Optimization]
    X -->J[Optimization]
    Y -->J[Optimization]
```

The network is a collection of vertices and edges. The vertices are connected by edges. The edges can be weighted, directed and/or weighted. The network can be used to find the optimal flow between two vertices. The flow is a path between two vertices. The flow is optimized by finding the path with the lowest weight.

\[   \left\{
\begin{array}{ll}
      0 & x\leq a \\
      \frac{x-a}{b-a} & a\leq x\leq b \\
      \frac{c-x}{c-b} & b\leq x\leq c \\
      1 & c\leq x \\
\end{array} 
\right. \]

\[\mathsf C : \qquad 
\begin{matrix}
&& \vdots &&\\
& \nearrow & s & \nwarrow &\\
x_0 & \longrightarrow & \vdots & \leftarrow & x_1 \\
& \searrow & s' & \swarrow &\\
&& \vdots & 
\end{matrix} \]

\[ \bullet  \swarrow \rightarrow \]

$$ \int $$
