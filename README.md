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